from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.vton.OmniTryVirtualTryOn import OmniTryVirtualTryOn
from torch.utils.data import Subset
import argparse
from pathlib import Path

def parseArgs():
    parser = argparse.ArgumentParser(description="Perform virtual try-on using OmniTryVirtualTryOn.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process. Not specified means all samples.")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained", help="Path to the pretrained models.")
    
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--should_quantize", default=False, action="store_true", help="Whether to quantize the model.")

    parser.add_argument("--offload", default=False, action="store_true", help="Enable model CPU offload.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Mixed precision dtype.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=int, nargs=2, default=[768, 1024], help="Size of the input images.")
    parser.add_argument("--guidance_scale", type=float, default=30.0, help="Image guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST
    
    dataset = TMPDataset(dataset_path_str, phase)
    if args.num_samples is not None and args.num_samples > 0:
        dataset = Subset(dataset, range(args.num_samples))

    flux_model_path = Path(args.pretrained_dir) / "flux-1" / "models-black-forest-labs-FLUX1-Fill-dev"
    if not flux_model_path.exists():
        flux_model_path = "black-forest-labs/FLUX.1-Fill-dev"
    
    lora_path = str(Path(args.pretrained_dir) / "omni_try" / "omnitry_v1_clothes.safetensors")
    
    omnitry_virtual_tryon = OmniTryVirtualTryOn(
        model_root=str(flux_model_path),
        lora_path=lora_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        offload=args.offload,
        mixed_precision=args.mixed_precision,
        should_quantize=args.should_quantize
    )
    
    omnitry_virtual_tryon.process(
        dataset,
        args.output_dir,
        n_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        size=tuple(args.size)
    )
