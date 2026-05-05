import os
import sys
from pathlib import Path
from PIL import Image
import torch

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
vendor_dir = os.path.join(root_dir, "vendors", "oot_diffusion")

if vendor_dir not in sys.path:
    sys.path.insert(0, vendor_dir)

from vendors.oot_diffusion.source.utils import get_mask_location
from vendors.oot_diffusion.source.preprocess.openpose.run_openpose import OpenPose
from vendors.oot_diffusion.source.preprocess.humanparsing.run_parsing import Parsing
from vendors.oot_diffusion.source.ootd.inference_ootd_dc import OOTDiffusionDC

from source.benchmark.dataset.TMPDataset import TMPDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

class OOTDiffusionVirtualTryOn:

    def __init__(
        self,
        model_root: str = "checkpoints",
        offload: bool = False,
        aggressive_offload: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        
        gpu_id = device.index if device.index is not None else 0
        if device.type == 'cpu':
             gpu_id = 0

        openpose_ckpts = os.path.join(model_root, "openpose", "ckpts")
        self.openpose_model = OpenPose(gpu_id, ckpts_path=openpose_ckpts)
        parsing_ckpts = os.path.join(model_root, "humanparsing")
        self.parsing_model = Parsing(gpu_id, ckpts_path=parsing_ckpts)
        self.model = OOTDiffusionDC(gpu_id, checkpoints_path=model_root)
        
        if offload:
            self.model.pipe.enable_model_cpu_offload()
        elif aggressive_offload:
            self.model.pipe.enable_sequential_cpu_offload()
            
    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        n_steps: int = 30,
        image_scale: float = 5.0,
        seed: int = 42,
        size: tuple[int, int] = (768, 1024)
    ):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        category_dict = ['upperbody', 'lowerbody', 'dress']
        category_dict_utils = ['upper_body', 'lower_body', 'dresses']

        print(f"Processing {len(dataset)} samples...")
        for sample in dataset:
            filename = sample["filename"]
            type_str = sample["type"]
            
            vton_img_path = str(sample["human_image"])
            garm_img_path = str(sample["vton_garment_path"])
            
            print(f"Processing {filename}...")
            
            is_dress = type_str == "female"
            category = 2 if is_dress else 0

            model_img = Image.open(vton_img_path).resize(size).convert('RGB')
            cloth_img = Image.open(garm_img_path).resize(size).convert('RGB')
            torch.cuda.empty_cache()
            
            keypoints = self.openpose_model(model_img.resize((int(size[0] / 2), int(size[1] / 2))))
            model_parse, _ = self.parsing_model(model_img.resize((int(size[0] / 2), int(size[1] / 2))))

            mask, mask_gray = get_mask_location("dc", category_dict_utils[category], model_parse, keypoints)
            mask = mask.resize(size, Image.NEAREST)
            mask_gray = mask_gray.resize(size, Image.NEAREST)
            
            masked_vton_img = Image.composite(mask_gray, model_img, mask)
            
            result_image = self.model(
                model_type="dc",
                category=category_dict[category],
                image_garm=cloth_img,
                image_vton=masked_vton_img,
                mask=mask,
                image_ori=model_img,
                num_samples=1,
                num_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
            )[0]
            
            generated_image_name = filename + ".jpg"
            full_output_path = os.path.join(output_dir, generated_image_name)
            
            result_image.save(full_output_path)
            print(f"Saved to {full_output_path}")
