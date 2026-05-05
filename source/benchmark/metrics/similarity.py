import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import lpips


def _apply_binary_mask(img_np: np.ndarray, mask_path: Optional[Path], target_size: tuple, inverted: bool = False) -> np.ndarray:
    if mask_path is None or not Path(mask_path).exists():
        return img_np

    mask = Image.open(mask_path).convert("L").resize(target_size)
    mask_np = np.where(np.array(mask) > 128, 1, 0).astype(np.uint8)
    if inverted:
        mask_np = 1 - mask_np
    mask_3c = np.stack([mask_np] * 3, axis=-1)
    return img_np * mask_3c


def _find_mask_for_image(image_path: Path, mask_dir: Path) -> Optional[Path]:
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    for ext in valid_extensions:
        candidate = mask_dir / (image_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def compute_ssim_pair(img1_path: Path, img2_path: Path,
                      mask1_path: Optional[Path] = None,
                      mask2_path: Optional[Path] = None,
                      inverted: bool = False) -> Optional[float]:
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    if img1 is None or img2 is None:
        print(f"Could not read: {img1_path.name} or {img2_path.name}")
        return None

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    target_size = (img1.shape[1], img1.shape[0])  # (width, height) for PIL

    if mask1_path or mask2_path:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_rgb = _apply_binary_mask(img1_rgb, mask1_path, target_size, inverted)
        img2_rgb = _apply_binary_mask(img2_rgb, mask2_path, target_size, inverted)
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=gray1.max() - gray1.min())


def compute_lpips_pair(img1_path: Path, img2_path: Path,
                       loss_fn: lpips.LPIPS,
                       mask1_path: Optional[Path] = None,
                       mask2_path: Optional[Path] = None,
                       inverted: bool = False) -> Optional[float]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)

    if mask1_path or mask2_path:
        img1_np = _apply_binary_mask(np.array(img1), mask1_path, img1.size, inverted)
        img2_np = _apply_binary_mask(np.array(img2), mask2_path, img1.size, inverted)
        img1 = Image.fromarray(img1_np.astype(np.uint8))
        img2 = Image.fromarray(img2_np.astype(np.uint8))

    tensor1 = transform(img1).unsqueeze(0)
    tensor2 = transform(img2).unsqueeze(0)

    device = next(loss_fn.parameters()).device
    with torch.no_grad():
        distance = loss_fn.forward(tensor1.to(device), tensor2.to(device))
    return distance.item()


def _mean_over_pairs(pairs, score_fn, desc: str) -> Optional[float]:
    total, count = 0.0, 0
    for pair in (pbar := __import__('tqdm').tqdm(pairs, desc=desc)):
        score = score_fn(pair)
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None
