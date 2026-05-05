import argparse
from pathlib import Path

from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.metrics.clip_t import compute_clip_t
from source.benchmark.metrics.fid import compute_fid
from source.benchmark.metrics.vton_similarity import (
    compute_cloth_ssim,
    compute_cloth_lpips,
    compute_identity_ssim,
    compute_identity_lpips,
)

def parseArgs():
    parser = argparse.ArgumentParser(description="Evaluate Virtual Try-On (VTON) results.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--inference_result_dir", type=str, required=True, help="Path to the inference results directory.")
    parser.add_argument("--frontal", action="store_true", help="Evaluate only frontal pose samples.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    inference_result_path = Path(args.inference_result_dir)
    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST

    dataset = TMPDataset(dataset_path_str, phase, frontal_only=args.frontal)

    print(f"FID: {compute_fid(dataset, inference_result_path)}")
    print(f"Cloth similarity:    LPIPS - {compute_cloth_lpips(dataset, inference_result_path)} | SSIM - {compute_cloth_ssim(dataset, inference_result_path)}")
    print(f"Identity preservation: LPIPS - {compute_identity_lpips(dataset, inference_result_path)} | SSIM - {compute_identity_ssim(dataset, inference_result_path)}")
