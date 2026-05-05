from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.vton.JcoVirtualTryOn import JcoVirtualTryOn
from torch.utils.data import Subset
import argparse
from pathlib import Path
import os

def parseArgs():
    parser = argparse.ArgumentParser(description="Perform virtual try-on using JcoVirtualTryOn.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process. Not specified means all samples.")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained", help="Path to the pretrained models.")
    parser.add_argument("--should_quantize", default=False, action="store_true", help="Whether to quantize the model.")

    parser.add_argument("--offload", default=False, action="store_true", help="Enable model CPU offload.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Mixed precision dtype.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=int, nargs=2, default=[768, 1024], help="Size of the input images.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Image guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST
    
    dataset = TMPDataset(dataset_path_str, phase)
    if args.num_samples is not None and args.num_samples > 0:
        dataset = Subset(dataset, range(args.num_samples))
    
    model_root = Path(args.pretrained_dir) / "jco_mvton"
    flux_model_path = Path(args.pretrained_dir) / "flux-1" / "models-black-forest-labs-FLUX1-dev"
    if not flux_model_path.exists():
        flux_model_path = "black-forest-labs/FLUX.1-dev"
    
    jco_virtual_tryon = JcoVirtualTryOn(
        model_root=str(flux_model_path),
        model_weight_dress=str(model_root / "try_on_dress.pt"),
        model_weight_upper=str(model_root / "try_on_upper.pt"),
        offload=args.offload,
        mixed_precision=args.mixed_precision,
        should_quantize=args.should_quantize
    )
    
    jco_virtual_tryon.process(
        dataset,
        args.output_dir,
        n_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        size=tuple(args.size)
    )
