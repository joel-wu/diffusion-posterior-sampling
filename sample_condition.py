from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from math import ceil


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    # logger
    logger = get_logger()

    SEED = 42
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    operator = cond_method.operator  # Get operator.forward

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    #sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    sample_fn = partial(
        sampler.p_sample_loop_beam_search,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        beam_width=8,
        topk=4,
        eta=0.5,
        selection_method="niqe"
    )

    # Working directory
    # out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    prefix = "test29"
    out_path = os.path.join(args.save_dir, f"{prefix}_{measure_config['operator']['name']}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception: In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )

    # Do Inference
    forward_inputs = []
    for i, ref_img in enumerate(loader):
        ref_img = ref_img.to(device)
        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn_i = partial(cond_method.conditioning, mask=mask)
            y = operator.forward(ref_img, mask=mask)
        else:
            mask = None
            measurement_cond_fn_i = measurement_cond_fn
            y = operator.forward(ref_img)

        y_n = noiser(y)

        forward_inputs.append({
            'ref_img': ref_img,
            'y_n': y_n,
            'mask': mask,
            'measurement_cond_fn': measurement_cond_fn_i,
            'fname': str(i).zfill(5) + '.png'
        })

    random.seed(None)
    np.random.seed(None)
    torch.manual_seed(torch.seed())
    torch.cuda.manual_seed_all(torch.seed())
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # batch_size = 4
    # num_batches = ceil(len(forward_inputs) / batch_size)

    for i, inp in enumerate(forward_inputs):
        x_start = torch.randn(inp['ref_img'].shape, device=device).requires_grad_()

        sample = sample_fn(
            x_start=x_start,
            measurement=inp['y_n'],
            record=True,
            save_root=out_path,
            measurement_cond_fn=inp['measurement_cond_fn'],
            operator=operator,
            mask=inp['mask']
        )

        # Save images
        plt.imsave(os.path.join(out_path, 'input', inp['fname']), clear_color(inp['y_n']))
        plt.imsave(os.path.join(out_path, 'label', inp['fname']), clear_color(inp['ref_img']))
        plt.imsave(os.path.join(out_path, 'recon', inp['fname']), clear_color(sample))



if __name__ == '__main__':
    main()
