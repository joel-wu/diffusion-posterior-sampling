import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from skvideo.measure import niqe
from tqdm.auto import tqdm

from util.img_utils import clear_color
import torchvision.transforms.functional as TF
from .posterior_mean_variance import get_mean_processor, get_var_processor

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):
    sampler = get_sampler(name=sampler)

    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]

    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised,
                   rescale_timesteps=rescale_timesteps)

def score_by_brightness(x0_pred,**kwargs):
    x0_clip = x0_pred.clamp(-1, 1)
    x0_norm = (x0_clip + 1) / 2
    R, G, B = x0_norm[:, 0:1], x0_norm[:, 1:2], x0_norm[:, 2:3]
    brightness_map = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return brightness_map.mean(dim=(1, 2, 3))  # shape: (B,)

def score_by_forward_residual(x0_pred, measurement, operator, **kwargs):
    Ax = operator.forward(x0_pred, **kwargs)
    diff = Ax - measurement
    return -torch.linalg.norm(diff.flatten(1), dim=1)

def score_by_consistency(x0_pred, x0_old,**kwargs):
    return -((x0_pred - x0_old) ** 2).mean(dim=(1, 2, 3))

def score_by_niqe(x0_pred, **kwargs):
    # Ensure range is in [-1, 1], NIQE expects this
    x0_clip = x0_pred.clamp(-1, 1)
    x0_norm = (x0_clip + 1) / 2  # to [0, 1]

    # Convert to grayscale using torchvision (standardized weights)
    # Input: (B, 3, H, W), Output: (B, 1, H, W)
    x0_gray = TF.rgb_to_grayscale(x0_norm, num_output_channels=1)

    x0_np = x0_gray.squeeze(1).detach().cpu().numpy()  # (B, H, W)

    scores = []
    for i in range(x0_np.shape[0]):
        img = (x0_np[i] * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        niqe_score = niqe(img)[0]
        scores.append(niqe_score)

    return -torch.tensor(scores, device=x0_pred.device, dtype=torch.float32)

def score_by_ensemble(x0_pred, measurement, operator, **kwargs):
    """
    Ensemble scoring combining NIQE and forward residual scores.
    Returns combined score where higher is better.
    """
    # Get NIQE scores (higher is better, so we negate the negative scores)
    niqe_scores = -score_by_niqe(x0_pred, **kwargs)
    
    # Get forward residual scores (higher is better, so we negate the negative scores)  
    forward_scores = -score_by_forward_residual(x0_pred, measurement, operator, **kwargs)
    
    # Normalize both scores to [0, 1] range for fair combination
    niqe_normalized = (niqe_scores - niqe_scores.min()) / (niqe_scores.max() - niqe_scores.min() + 1e-8)
    forward_normalized = (forward_scores - forward_scores.min()) / (forward_scores.max() - forward_scores.min() + 1e-8)
    
    # Combine scores (equal weight)
    ensemble_scores = 0.5 * niqe_normalized + 0.5 * forward_normalized
    
    return ensemble_scores

def score_by_bandpass_diff(x0_pred, x0_old, progress):

    def highpass_filter(x, cutoff=0.25):
        """
        Apply high-pass filter by zeroing out low-frequency FFT components.
        cutoff: fraction of frequency (e.g., 0.25 means remove lowest 25%)
        """
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        # Create a circular mask that keeps only high frequencies
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        rr = torch.sqrt(xx ** 2 + yy ** 2).to(x.device)
        mask = (rr >= cutoff).float()

        # Broadcast to batch and channel
        mask = mask[None, None, :, :].expand_as(x_fft)
        x_fft_filtered = x_fft * mask

        # Inverse FFT
        x_filtered = torch.fft.ifftshift(x_fft_filtered)
        x_out = torch.fft.ifft2(x_filtered, norm="ortho").real
        return x_out

    def bandpass_filter(x, f_low=0.2, f_high=0.4):
        """
        Apply band-pass filter by masking the FFT.
        f_low and f_high ∈ [0, 1], representing relative radius in normalized freq
        """
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        rr = torch.sqrt(xx ** 2 + yy ** 2).to(x.device)
        mask = ((rr >= f_low) & (rr < f_high)).float()

        mask = mask[None, None, :, :].expand_as(x_fft)
        x_fft_filtered = x_fft * mask

        x_filtered = torch.fft.ifftshift(x_fft_filtered)
        x_out = torch.fft.ifft2(x_filtered, norm="ortho").real
        return x_out

    x0_band = bandpass_filter(x0_pred, progress, min(progress + 0.2, 1.0))
    xold_band = bandpass_filter(x0_old, progress, min(progress + 0.2, 1.0))
    return -((x0_band - xold_band) ** 2).mean(dim=(1, 2, 3))



class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <= 1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)

        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        # noise = torch.randn_like(x_start)
        gen = torch.Generator(device=x_start.device)
        gen.manual_seed(42 + t[0].item())
        noise = torch.randn(x_start.shape, dtype=x_start.dtype, device=x_start.device, generator=gen)

        assert noise.shape == x_start.shape

        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root):
        """
        The function used for sampling from noise.
        """
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)

            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)

            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                                measurement=measurement,
                                                noisy_measurement=noisy_measurement,
                                                x_prev=img,
                                                x_0_hat=out['pred_xstart'])
            img = img.detach_()

            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img


    def p_sample_loop_beam_search(
            self,
            model,
            x_start,
            measurement,
            measurement_cond_fn,
            operator,
            record,
            save_root,
            beam_width=2,
            topk=1,
            eta=0.5,
            search_strategy="beam", 
            selection_method="brightness",
            **kwargs
    ):
        SELECTION_FN_REGISTRY = {
            "brightness": score_by_brightness,
            "consistency": score_by_consistency,
            "forward": score_by_forward_residual,
            "niqe": score_by_niqe,
            "ensemble": score_by_ensemble,
        }

        selection_fn = SELECTION_FN_REGISTRY[selection_method]


        device = x_start.device

        if search_strategy == "bon":
            x_prev = torch.randn((beam_width,) + x_start.shape[1:], device=device).contiguous().requires_grad_(True)
        else:
            x_prev = x_start.repeat(beam_width, 1, 1, 1).contiguous().detach().requires_grad_(True)


        selection_interval = 50 if search_strategy == "beam" else self.num_timesteps + 1


        x0_old = None

        for idx in tqdm(list(range(self.num_timesteps))[::-1]):

            #print(idx)

            t = torch.full((beam_width,), idx, device=device, dtype=torch.long)

            # Forward pass

            # print(f"[DEBUG] x_prev.shape = {x_prev.shape}")
            model_out = model(x_prev, self._scale_timesteps(t))
            if model_out.shape[1] == 2 * x_prev.shape[1]:
                model_out, model_var_values = torch.chunk(model_out, 2, dim=1)
            else:
                model_var_values = model_out

            model_mean, x0_pred = self.mean_processor.get_mean_and_xstart(x_prev, t, model_out)

            eps = self.predict_eps_from_x_start(x_prev, t, x0_pred)

            if idx != self.num_timesteps - 1 and idx % selection_interval == 0:

                scores = selection_fn(
                    x0_pred=x0_pred,
                    x0_old=x0_old,
                    measurement=measurement,
                    operator=operator,
                    progress=1.0 - idx / self.num_timesteps,
                    **kwargs
                )
                topk_indices = torch.topk(scores, topk).indices

                x_prev = x_prev[topk_indices]

                x_prev = x_prev.repeat_interleave(beam_width // topk, dim=0)

                model_out = model(x_prev, self._scale_timesteps(t))
                if model_out.shape[1] == 2 * x_prev.shape[1]:
                    model_out, model_var_values = torch.chunk(model_out, 2, dim=1)
                else:
                    model_var_values = model_out

                model_mean, x0_pred = self.mean_processor.get_mean_and_xstart(x_prev, t, model_out)

                x0_old = x0_pred


            alpha_bar = extract_and_expand(self.alphas_cumprod, t, x_prev)
            alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x_prev)
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
                    torch.sqrt(1 - alpha_bar / alpha_bar_prev)

            noise = torch.randn_like(x_prev)  # shape: (beam_width, C, H, W)

            mean_pred = x0_pred * torch.sqrt(alpha_bar_prev) + \
                        torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

            if idx != 0:
                x_next = mean_pred + sigma * noise

            x_next, _ = measurement_cond_fn(x_t=x_next, measurement=measurement, x_prev=x_prev, x_0_hat=x0_pred)

            x_prev = x_next.detach().requires_grad_(True)

            if idx == self.num_timesteps - 1:

                x0_old = x0_pred


            # x_prev = x_prev.detach()
            x_prev_vis = x_prev.detach()

            if record and idx % 10 == 0:
                os.makedirs(os.path.join(save_root, "progress"), exist_ok=True)
                plt.imsave(
                    os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png"),
                    clear_color(x_prev_vis[0])
                )

        final_scores = selection_fn(
            x0_pred=x0_pred,
            x0_old=x0_old,
            measurement=measurement,
            operator=operator,
            progress=0.0,
            **kwargs
        )

        best_idx = torch.argmax(final_scores)  # or argmin
        x0_best = x0_pred[best_idx]

        # 保存所有x0_pred到all_result文件夹
        all_result_dir = os.path.join(save_root, "all_result")
        os.makedirs(all_result_dir, exist_ok=True)
        for i in range(x0_pred.shape[0]):
            filename = f"x0_pred_{i:02d}.png"
            plt.imsave(
                os.path.join(all_result_dir, filename),
                clear_color(x0_pred[i])
            )

        return x0_best

    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong.
            # This will just provide shape information, and
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}


@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t, eta=0.0):
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])

    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
