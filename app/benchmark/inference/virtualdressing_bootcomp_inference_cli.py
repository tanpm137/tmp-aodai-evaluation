from source.benchmark.dataset.TMPDataset import TMPDataset
from source.benchmark.virtual_dressing.BootcompVirtualDressing import BootcompVirtualDressing
from torch.utils.data import Subset
import argparse
from pathlib import Path

def parseArgs():
    parser = argparse.ArgumentParser(description="Perform virtual dressing using BootcompVirtualDressing.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process. Not specified means all samples.")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained", help="Path to the pretrained models.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=int, nargs=2, default=[576, 768], help="Size of the input images.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for the model.")
    parser.add_argument("--cloth_scale", type=float, default=2.0, help="Cloth scale for the model.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision for the model.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()

    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST
    
    dataset = TMPDataset(dataset_path_str, phase)
    if args.num_samples is not None and args.num_samples > 0:
        dataset = Subset(dataset, range(args.num_samples))
    
    pretrained_dir = Path(args.pretrained_dir) / "bootcomp"

    if pretrained_dir.exists():
        bootcomp_dressing = BootcompVirtualDressing(
            pretrained_model_name_or_path=pretrained_dir / "models-SG161222-realvisxl-v3.0",
            unet_encoder_ckpt=pretrained_dir / "ckpt-omniousai-bootcomp",
            pretrained_vae_path=pretrained_dir / "models-madebyollin-sdxl-vae-fp16-fix",
            mixed_precision=args.mixed_precision
        )
    else:
        bootcomp_dressing = BootcompVirtualDressing(mixed_precision=args.mixed_precision)

    bootcomp_dressing.process(
        dataset,
        args.output_dir,
        guidance_scale=args.guidance_scale,
        cloth_scale=args.cloth_scale,
        num_inference_steps=args.num_inference_steps,
        size=args.size,
        seed=args.seed
    )


