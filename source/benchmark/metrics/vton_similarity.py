from pathlib import Path
from typing import Optional

import lpips
import torch
from tqdm import tqdm

from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.metrics.similarity import (
    compute_ssim_pair,
    compute_lpips_pair,
    _find_mask_for_image,
)


def _build_vton_pairs(dataset: TMPDataset, inference_result_path: Path):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    available_generated = {
        p.stem: p for p in inference_result_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    }
    cloth_mask_dir = inference_result_path / "cloth_masks"

    pairs = []
    for sample in dataset:
        filename = sample["filename"]
        generated_path = available_generated.get(filename)
        if generated_path is None:
            print(f"Not found generated image: {filename}")
            continue

        vton_compared = sample.get("vton_compared_path")
        if vton_compared is None or not Path(vton_compared).exists():
            print(f"Not found vton_compared_path for: {filename}")
            continue

        human_image = sample.get("human_image")
        if human_image is None or not Path(human_image).exists():
            print(f"Not found human_image for: {filename}")
            continue

        vton_compared = Path(vton_compared)
        human_image = Path(human_image)

        gen_cloth_mask = _find_mask_for_image(generated_path, cloth_mask_dir) if cloth_mask_dir.exists() else None
        gt_cloth_mask = None
        human_mask_path = sample.get("human_mask")
        if human_mask_path:
            dataset_cloth_mask_dir = Path(human_mask_path).parent.parent / "cloth_masks"
            gt_cloth_mask = _find_mask_for_image(vton_compared, dataset_cloth_mask_dir)

        pairs.append({
            "generated": generated_path,
            "vton_compared": vton_compared,
            "human_image": human_image,
            "gen_cloth_mask": gen_cloth_mask,
            "gt_cloth_mask": gt_cloth_mask,
            "human_cloth_mask": sample.get("cloth_mask"),
        })

    return pairs


def compute_cloth_ssim(dataset: TMPDataset, inference_result_path: Path) -> Optional[float]:
    """SSIM on cloth region: generated vs vton_compared_path, both masked by cloth_mask."""
    print("Calculating Cloth SSIM (VTON)")
    pairs = _build_vton_pairs(dataset, inference_result_path)
    if not pairs:
        return None

    total, count = 0.0, 0
    for pair in tqdm(pairs, desc="Cloth SSIM"):
        score = compute_ssim_pair(
            img1_path=pair["generated"],
            img2_path=pair["vton_compared"],
            mask1_path=pair["gen_cloth_mask"],
            mask2_path=pair["gt_cloth_mask"],
        )
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None


def compute_cloth_lpips(dataset: TMPDataset, inference_result_path: Path,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu",
                         net: str = "alex") -> Optional[float]:
    """LPIPS on cloth region: generated vs vton_compared_path, both masked by cloth_mask."""
    print(f"Calculating Cloth LPIPS (VTON) on {device}")
    pairs = _build_vton_pairs(dataset, inference_result_path)
    if not pairs:
        return None

    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    total, count = 0.0, 0
    for pair in tqdm(pairs, desc="Cloth LPIPS"):
        score = compute_lpips_pair(
            img1_path=pair["generated"],
            img2_path=pair["vton_compared"],
            loss_fn=loss_fn,
            mask1_path=pair["gen_cloth_mask"],
            mask2_path=pair["gt_cloth_mask"],
        )
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None


def compute_identity_ssim(dataset: TMPDataset, inference_result_path: Path) -> Optional[float]:
    """SSIM on non-cloth (identity) region: generated vs human_image, both masked by INVERTED cloth_mask."""
    print("Calculating Identity SSIM (VTON)")
    pairs = _build_vton_pairs(dataset, inference_result_path)
    if not pairs:
        return None

    total, count = 0.0, 0
    for pair in tqdm(pairs, desc="Identity SSIM"):
        score = compute_ssim_pair(
            img1_path=pair["generated"],
            img2_path=pair["human_image"],
            mask1_path=pair["gen_cloth_mask"],
            mask2_path=pair["human_cloth_mask"],
            inverted=True,
        )
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None


def compute_identity_lpips(dataset: TMPDataset, inference_result_path: Path,
                            device: str = "cuda" if torch.cuda.is_available() else "cpu",
                            net: str = "alex") -> Optional[float]:
    """LPIPS on non-cloth (identity) region: generated vs human_image, both masked by INVERTED cloth_mask."""
    print(f"Calculating Identity LPIPS (VTON) on {device}")
    pairs = _build_vton_pairs(dataset, inference_result_path)
    if not pairs:
        return None

    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    total, count = 0.0, 0
    for pair in tqdm(pairs, desc="Identity LPIPS"):
        score = compute_lpips_pair(
            img1_path=pair["generated"],
            img2_path=pair["human_image"],
            loss_fn=loss_fn,
            mask1_path=pair["gen_cloth_mask"],
            mask2_path=pair["human_cloth_mask"],
            inverted=True,
        )
        if score is not None:
            total += score
            count += 1
    return total / count if count > 0 else None
