import argparse
import os
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from torch.utils.data import Subset

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from source.benchmark.dataset.TMPDataset import TMPDataset

GEMINI_MODEL = "gemini-2.5-flash-image-preview"
PROMPT = """
The resulted image must be in resolution of {width}x{height}.
Do not include any introductory phrases or conversational filler. Begin your response directly with the resulted image itself.
"""

def parseArgs():
    parser = argparse.ArgumentParser(description="Perform virtual dressing using Gemini.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process. Not specified means all samples.")
    parser.add_argument("--size", type=int, nargs=2, default=[576, 768], help="Size of the input images.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds.")
    
    return parser.parse_args()

def main():
    args = parseArgs()

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise Exception("ERROR: GEMINI_API_KEY environment variable is not set. Please set it or ensure your environment is authenticated.")

    print("Setting up...")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_path_str = args.dataset_dir
    phase: TMPDataset.Phase = TMPDataset.Phase.TEST
    
    dataset = TMPDataset(dataset_path_str, phase)
    if args.num_samples is not None and args.num_samples > 0:
        dataset = Subset(dataset, range(args.num_samples))
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    generation_config = GenerationConfig()

    width, height = args.size
    formatted_prompt = PROMPT.format(width=width, height=height)

    print(f"Processing {len(dataset)} samples...")
    for sample in tqdm(dataset, "Processing"):
        filename = sample["filename"]
        pose_description = sample["pose_description"]
        cloth_path = sample["garment_path"]

        text_prompt = pose_description + formatted_prompt
        try:
            garment_image = Image.open(cloth_path)
        except FileNotFoundError:
            print(f"\nError: The file at {cloth_path} was not found.")
            continue
        except Exception as e:
            print(f"\nAn error occurred while opening the image: {e}")
            continue
        
        output_name = filename + ".jpg"
        
        try:
            response = model.generate_content(
                [text_prompt, garment_image],
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                generated_image_part = response.candidates[0].content.parts[0]
                if generated_image_part.inline_data:
                    image_data = generated_image_part.inline_data.data
                    output_image = Image.open(BytesIO(image_data))
                    output_image.save(os.path.join(output_dir, output_name))
                else:
                    print(f"\nError: Could not find inline image data for {output_name} in the API response.")
                    error_file_path = os.path.join(output_dir, filename + ".txt")
                    with open(error_file_path, "w") as f:
                        f.write(str(response))
            else:
                print(f"\nError: Invalid API response format for {output_name}.")
                error_file_path = os.path.join(output_dir, filename + ".txt")
                with open(error_file_path, "w") as f:
                    f.write(str(response))
            
            time.sleep(args.delay)
            
        except Image.UnidentifiedImageError:
            print(f"Error: Cannot identify image {output_name} file. It might be corrupted.")
            continue
        except Exception as e:
            print(f"\nAn unexpected error occurred during the API request for {output_name}: {e}")
            continue

if __name__ == "__main__":
    main()
