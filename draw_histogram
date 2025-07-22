import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch
from skvideo.measure import niqe
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr
from skimage.transform import resize
from guided_diffusion.measurements import get_operator
import yaml
import random
from util.img_utils import mask_generator
from guided_diffusion.measurements import get_noise
import torchvision.utils as vutils
from torchvision import transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load task_config
with open('/home/ubuntu/LLM-inference/yuheng-project/diff/diffusion-posterior-sampling/configs/inpainting_config.yaml') as f:
    task_config = yaml.load(f, Loader=yaml.FullLoader)
measure_config = task_config['measurement']

# Initialize operator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
operator = get_operator(device=device, **measure_config['operator'])

# Load mask_opt
mask_opt = measure_config['mask_opt']
mask_gen = mask_generator(**mask_opt)

# Initialize noiser
noiser = get_noise(**measure_config['noise'])

# 1. Path settings
base_dir = '/home/ubuntu/LLM-inference/yuheng-project/diff/diffusion-posterior-sampling/results/DPS_BoN_PSNR_inpainting'
all_result_dir = os.path.join(base_dir, 'all_result')
label_img_path = os.path.join(base_dir, 'label', '00000.png')
y_img_path = os.path.join(base_dir, 'input', '00000.png')

all_imgs = sorted(glob.glob(os.path.join(all_result_dir, 'x0_pred_*.png')))
print(f"Found {len(all_imgs)} images in all_result.")

# Read label/00000.png
ref_img = imread(label_img_path)
if ref_img.ndim == 2:
    ref_img = np.stack([ref_img]*3, axis=-1)
if ref_img.shape[-1] == 4:
    ref_img = ref_img[..., :3]
ref_img = ref_img.astype(np.float32) / 255.0

# 3. Load y (input/00000.png)
y_img = imread(y_img_path)
if y_img.ndim == 2:
    y_img = np.stack([y_img]*3, axis=-1)
if y_img.shape[-1] == 4:
    y_img = y_img[..., :3]
y_img = y_img.astype(np.float32) / 255.0
# Convert to torch tensor, simulating the input format of the inpainting operator
# Assuming y_img is [H,W,3], convert to [1,3,H,W]
y_tensor = torch.from_numpy(y_img.transpose(2,0,1)).unsqueeze(0).float()

psnr_list = []
niqe_list = []
residual_list = []
img_tensors = []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
ref_img_tensor = transform(ref_img).unsqueeze(0).to(device)  # [1, 3, H, W]
mask = mask_gen(ref_img_tensor)
mask = mask[:, 0, :, :].unsqueeze(dim=0)

# Generate y and y_n
y = operator.forward(ref_img_tensor, mask=mask)
y_n = noiser(y)
y_tensor = y_n  # Use noisy observation

# Save y and y_n as images for easy inspection
def save_tensor_img(tensor, path):
    # tensor: [1, 3, H, W] or [3, H, W]
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy().transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to 0-1
    plt.imsave(path, img)

save_tensor_img(y, 'check_y.png')
save_tensor_img(y_n, 'check_y_n.png')

for img_path in all_imgs:
    img = imread(img_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.float32) / 255.0
    if img.shape != ref_img.shape:
        from skimage.transform import resize
        img = resize(img, ref_img.shape, order=1, preserve_range=True, anti_aliasing=True)
        img = np.clip(img, 0, 1)
    psnr = compare_psnr(ref_img, img, data_range=1.0)
    psnr_list.append(psnr)
    img_tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)  # [1, 3, H, W]
    img_tensor = img_tensor.clamp(0, 1)
    img_gray = TF.rgb_to_grayscale(img_tensor, num_output_channels=1)
    img_np = img_gray.squeeze(0).squeeze(0).detach().cpu().numpy()  # (H, W)
    img_np = (img_np * 255).astype(np.uint8)
    img_np = np.expand_dims(img_np, axis=0)  # (1, H, W)
    niqe_score = niqe(img_np)[0]
    niqe_list.append(niqe_score)
    with torch.no_grad():
        Ax = operator.forward(img_tensor.to(device), mask=mask)
        diff = Ax - y_tensor.to(device)
        residual = torch.norm(diff.flatten(), p=2).item()
    residual_list.append(residual)
    img_tensors.append(img_tensor)

psnr_arr = np.array(psnr_list)
niqe_arr = np.array(niqe_list)
residual_arr = np.array(residual_list)

corr, pval = pearsonr(psnr_arr, niqe_arr)
print(f"Pearson correlation (PSNR vs NIQE): {corr:.3f}, p-value: {pval:.3g}")

# PSNR vs Residual correlation (PSNR >= 15)
mask_valid = psnr_arr >= 15
psnr_arr_valid = psnr_arr[mask_valid]
residual_arr_valid = residual_arr[mask_valid]
if len(psnr_arr_valid) > 1:
    corr_res, pval_res = pearsonr(psnr_arr_valid, residual_arr_valid)
    print(f"Pearson correlation (PSNR vs Residual, PSNR>=15): {corr_res:.3f}, p-value: {pval_res:.3g}")
else:
    print("Not enough valid points for PSNR vs Residual correlation (PSNR>=15)")

# PSNR histogram
plt.figure()
plt.hist(psnr_arr, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('PSNR')
plt.ylabel('Count')
plt.title('Histogram of PSNR')
plt.savefig('psnr_histogram.png')
plt.close()

# Residual histogram
plt.figure()
plt.hist(residual_arr, bins=30, color='salmon', edgecolor='black')
plt.xlabel('Residual ||y - A(x)||_2')
plt.ylabel('Count')
plt.title('Histogram of Forward Residual')
plt.savefig('residual_histogram.png')
plt.close()

# PSNR vs NIQE scatter
plt.figure()
plt.scatter(psnr_arr, niqe_arr, alpha=0.7, label='Samples')
if len(psnr_arr) > 1:
    z = np.polyfit(psnr_arr, niqe_arr, 1)
    p = np.poly1d(z)
    psnr_fit = np.linspace(psnr_arr.min(), psnr_arr.max(), 100)
    plt.plot(psnr_fit, p(psnr_fit), 'r--', label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
plt.xlabel('PSNR')
plt.ylabel('NIQE')
plt.title('PSNR vs NIQE')
plt.legend()
plt.savefig('psnr_vs_niqe.png')
plt.close()

# PSNR vs Residual scatter (PSNR >= 15)
plt.figure()
plt.scatter(psnr_arr_valid, residual_arr_valid, alpha=0.7, label='Samples')
if len(psnr_arr_valid) > 1:
    z = np.polyfit(psnr_arr_valid, residual_arr_valid, 1)
    p = np.poly1d(z)
    psnr_fit = np.linspace(psnr_arr_valid.min(), psnr_arr_valid.max(), 100)
    plt.plot(psnr_fit, p(psnr_fit), 'r--', label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
plt.xlabel('PSNR')
plt.ylabel('Residual')
plt.title('PSNR vs Residual (PSNR>=15)')
plt.legend()
plt.savefig('psnr_vs_residual_filtered.png')
plt.close()

# Ensemble selection histogram
ensemble_scores = []
group_psnr = []
group_size = 8
for i in range(0, len(psnr_arr), group_size):
    group_niqe = niqe_arr[i:i+group_size]
    group_residual = residual_arr[i:i+group_size]
    niqe_norm = (group_niqe - group_niqe.min()) / (group_niqe.max() - group_niqe.min() + 1e-8)
    residual_norm = (group_residual - group_residual.min()) / (group_residual.max() - group_residual.min() + 1e-8)
    ensemble = niqe_norm - residual_norm
    idx = np.argmax(ensemble)
    ensemble_scores.append(ensemble[idx])
    group_psnr.append(psnr_arr[i + idx])

plt.figure()
plt.hist(group_psnr, bins=30, color='limegreen', edgecolor='black')
plt.xlabel('PSNR (ensemble selected)')
plt.ylabel('Count')
plt.title('Histogram of PSNR (Ensemble Selected)')
plt.savefig('ensemble_psnr_histogram.png')
plt.close()

print(f"Mean PSNR: {psnr_arr.mean():.3f}")
print(f"Mean NIQE: {niqe_arr.mean():.3f}")
print(f"Mean Residual: {residual_arr.mean():.3f}")
print(f"Mean Ensemble-Selected PSNR: {np.mean(group_psnr):.3f}")

# Print max PSNR image index
max_psnr_idx = np.argmax(psnr_arr)
print(f"Max PSNR image index: {max_psnr_idx}, PSNR: {psnr_arr[max_psnr_idx]:.3f}")
