from pathlib import Path
from typing import Optional

import lpips
import torch
from tqdm import tqdm

from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.metrics.similarity import compute_ssim_pair, compute_lpips_pair


def _build_virtual_dressing_pairs(dataset: TMPDataset, inference_result_path: Path):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    available_generated = {
        p.stem: p for p in inference_result_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    }

    cloth_mask_dir = inference_result_path / "cloth_masks"
    available_gen_cloth_masks = {
        m.stem: m for m in cloth_mask_dir.iterdir()
        if m.suffix.lower() in valid_extensions
    } if cloth_mask_dir.exists() else {}

    human_mask_dir = inference_result_path / "human_masks"
    available_gen_human_masks = {
        m.stem: m for m in human_mask_dir.iterdir()
        if m.suffix.lower() in valid_extensions
    } if human_mask_dir.exists() else {}

    pairs = []
    for sample in dataset:
        filename = sample["filename"]
        generated_path = available_generated.get(filename)
        if generated_path is None:
            print(f"Not found generated image: {filename}")
            continue

        human_image = sample.get("human_image")
        if human_image is None or not Path(human_image).exists():
            print(f"Not found human_image for: {filename}")
            continue

        pairs.append({
            "generated": generated_path,
            "human_image": Path(human_image),
            "gen_cloth_mask": available_gen_cloth_masks.get(filename),
            "gt_cloth_mask": sample.get("cloth_mask"),
            "gen_human_mask": available_gen_human_masks.get(filename),
            "gt_human_mask": sample.get("human_mask"),
        })

    return pairs


def _run_metric(pairs, score_fn, desc: str) -> Optional[float]:
    total, count = 0.0, 0
    for pair in tqdm(pairs, desc=desc):
        score = score_fn(pair)
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None


def compute_cloth_ssim(dataset: TMPDataset, inference_result_path: Path) -> Optional[float]:
    """SSIM on cloth region: generated vs human_image, both apply cloth_mask."""
    print("Calculating Cloth SSIM (Virtual Dressing)")
    pairs = _build_virtual_dressing_pairs(dataset, inference_result_path)
    if not pairs:
        return None
    return _run_metric(pairs, lambda p: compute_ssim_pair(
        img1_path=p["generated"],
        img2_path=p["human_image"],
        mask1_path=p["gen_cloth_mask"],
        mask2_path=p["gt_cloth_mask"],
    ), desc="Cloth SSIM")


def compute_cloth_lpips(dataset: TMPDataset, inference_result_path: Path,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu",
                         net: str = "alex") -> Optional[float]:
    """LPIPS on cloth region: generated vs human_image, both apply cloth_mask."""
    print(f"Calculating Cloth LPIPS (Virtual Dressing) on {device}")
    pairs = _build_virtual_dressing_pairs(dataset, inference_result_path)
    if not pairs:
        return None
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    return _run_metric(pairs, lambda p: compute_lpips_pair(
        img1_path=p["generated"],
        img2_path=p["human_image"],
        loss_fn=loss_fn,
        mask1_path=p["gen_cloth_mask"],
        mask2_path=p["gt_cloth_mask"],
    ), desc="Cloth LPIPS")


def compute_person_ssim(dataset: TMPDataset, inference_result_path: Path) -> Optional[float]:
    """SSIM on person region: generated vs human_image, both apply human_mask."""
    print("Calculating Person SSIM (Virtual Dressing)")
    pairs = _build_virtual_dressing_pairs(dataset, inference_result_path)
    if not pairs:
        return None
    return _run_metric(pairs, lambda p: compute_ssim_pair(
        img1_path=p["generated"],
        img2_path=p["human_image"],
        mask1_path=p["gen_human_mask"],
        mask2_path=p["gt_human_mask"],
    ), desc="Person SSIM")


def compute_person_lpips(dataset: TMPDataset, inference_result_path: Path,
                          device: str = "cuda" if torch.cuda.is_available() else "cpu",
                          net: str = "alex") -> Optional[float]:
    """LPIPS on person region: generated vs human_image, both apply human_mask."""
    print(f"Calculating Person LPIPS (Virtual Dressing) on {device}")
    pairs = _build_virtual_dressing_pairs(dataset, inference_result_path)
    if not pairs:
        return None
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    return _run_metric(pairs, lambda p: compute_lpips_pair(
        img1_path=p["generated"],
        img2_path=p["human_image"],
        loss_fn=loss_fn,
        mask1_path=p["gen_human_mask"],
        mask2_path=p["gt_human_mask"],
    ), desc="Person LPIPS")
