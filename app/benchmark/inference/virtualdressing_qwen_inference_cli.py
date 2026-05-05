import argparse
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from pathlib import Path
from torch.utils.data import Subset

from source.benchmark.dataset.TMPDataset import TMPDataset

def parseArgs():
    parser = argparse.ArgumentParser(description="Perform virtual dressing using QwenImageEditPipeline.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process. Not specified means all samples.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=int, nargs=2, default=[576, 768], help="Size of the input images.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for the model.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()

    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST
    
    dataset = TMPDataset(dataset_path_str, phase)
    if args.num_samples is not None and args.num_samples > 0:
        dataset = Subset(dataset, range(args.num_samples))
    
    result_path = args.output_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    torch_dtype = torch.bfloat16
    device = "cuda:0"

    print("Loading Qwen-Image-Edit pipeline...")
    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch_dtype,
    )
    pipeline.to(device)
    print("Pipeline loaded successfully.")

    positive_prompt = ". Best quality, high quality, basic background, modest, realistic"
    negative_prompt = "bare, monochrome, lowres, bad anatomy, worst quality, low quality"

    width, height = args.size
    image_size = (width, height)

    print(f"Processing {len(dataset)} samples...")
    for sample in dataset:
        filename = sample["filename"]
        print(f"Processing {filename}...")

        pose_promt = sample["pose_description"]
        cloth_path = sample["garment_path"]

        cloth_image = Image.open(cloth_path).convert("RGB").resize(image_size)
        prompt = pose_promt + positive_prompt
            
        inputs = {
            "image": cloth_image,
            "prompt": prompt,
            "generator": torch.manual_seed(args.seed),
            "true_cfg_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "negative_prompt": negative_prompt,
        }
        
        try:
            with torch.inference_mode():
                output_image = pipeline(**inputs).images[0]

            output_name = filename + ".jpg"
            output_file_path = os.path.join(result_path, output_name)
            output_image.save(output_file_path)
            print(f"Save completed: {output_file_path}")

        except Exception as e:
            print(f"An error occurred during the pipeline execution for {filename}: {e}")
