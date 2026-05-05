from vendors.fitdit.source.pose_guider import PoseGuider
import os
import math
import sys
import pandas as pd
from pathlib import Path
import random
import csv

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
vendor_dir = os.path.join(root_dir, "vendors", "fitdit")

if vendor_dir not in sys.path:
    sys.path.insert(0, vendor_dir)

from vendors.fitdit.source.preprocess.humanparsing.run_parsing import Parsing
from vendors.fitdit.source.preprocess.dwpose import DWposeDetector
from vendors.fitdit.source.utils_mask import get_mask_location
from vendors.fitdit.source.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from vendors.fitdit.source.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from vendors.fitdit.source.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton

from source.benchmark.dataset.TMPDataset import TMPDataset

class FitDitVirtualTryOn:

    def __init__(self, 
                 model_root: str, 
                 offload: bool = False, 
                 aggressive_offload: bool = False, 
                 mixed_precision: str = "fp16",
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        device_str = "cuda" if device.type == "cuda" else "cpu"
        
        match mixed_precision:
            case "fp16":
                weight_dtype = torch.float16
            case "bf16":
                weight_dtype = torch.bfloat16
            case _:
                weight_dtype = torch.float32
        
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype)
        pose_guider =  PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider.load_state_dict(torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin")))
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
        pose_guider.to(device=device_str, dtype=weight_dtype)
        image_encoder_large.to(device=device_str)
        image_encoder_bigG.to(device=device_str)
        
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(model_root, torch_dtype=weight_dtype, transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, image_encoder_large=image_encoder_large, image_encoder_bigG=image_encoder_bigG)
        self.pipeline.to(device_str)
        
        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.pipeline.to(device_str)
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device_str)
            self.parsing_model = Parsing(model_root=model_root, device=device_str)

    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        n_steps: int = 30,
        image_scale: float = 5.0,
        seed: int = 43,
        size: tuple[int, int] = (768, 1024)
    ):
        num_images_per_prompt: int = 1
        new_width, new_height = size
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {len(dataset)} samples...")
        for sample in dataset:
            filename = sample["filename"]
            print(f"Processing {filename}...")
            
            type_str = sample["type"]
            category = "Dresses" if type_str == "female" else "Upper-body"
            
            vton_img_path = str(sample["human_image"])
            garm_img_path = str(sample["vton_garment_path"])
            
            mask_vton_img, pose_image = self._generate_mask(vton_img_path, category, 0, 0, 0, 0)

            res = self._process_single(
                vton_img=vton_img_path, 
                garm_img=garm_img_path, 
                pre_mask=mask_vton_img, 
                pose_image=pose_image, 
                n_steps=n_steps, 
                image_scale=image_scale, 
                seed=seed, 
                num_images_per_prompt=num_images_per_prompt, 
                new_width=new_width, 
                new_height=new_height
            )
            
            result_img = res[0]
            generated_image_name = filename + ".jpg"
            full_output_path = os.path.join(output_dir, generated_image_name)
            
            result_img.save(full_output_path)
            print(f"Saved to {full_output_path}")

    def _generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img = Image.open(vton_img).convert('RGB')
            vton_img_det = self._resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height

            pose_image = pose_image[:,:,::-1] #rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {}
            im['background'] = np.array(vton_img.convert("RGBA"))
            im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)]
            im['composite'] = np.array(masked_vton_img.convert("RGBA"))
            
            return im, pose_image

    def _process_single(self, vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images_per_prompt, new_width, new_height):
        with torch.inference_mode():
            garm_img = Image.open(garm_img)
            vton_img = Image.open(vton_img)

            model_image_size = vton_img.size
            garm_img, _, _ = self._pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = self._pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

            mask = pre_mask["layers"][0][:,:,3]
            mask = Image.fromarray(mask)
            mask, _, _ = self._pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            mask = mask.convert("L")
            pose_image, _, _ = self._pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            if seed==-1:
                seed = random.randint(0, 2147483647)
            res = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images_per_prompt
            ).images
            for idx in range(len(res)):
                res[idx] = self._unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
            return res


    def _pad_and_resize(self, im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
        old_width, old_height = im.size
        
        ratio_w = new_width / old_width
        ratio_h = new_height / old_height
        if ratio_w < ratio_h:
            new_size = (new_width, round(old_height * ratio_w))
        else:
            new_size = (round(old_width * ratio_h), new_height)
        
        im_resized = im.resize(new_size, mode)

        pad_w = math.ceil((new_width - im_resized.width) / 2)
        pad_h = math.ceil((new_height - im_resized.height) / 2)

        new_im = Image.new('RGB', (new_width, new_height), pad_color)
        
        new_im.paste(im_resized, (pad_w, pad_h))

        return new_im, pad_w, pad_h

    def _unpad_and_resize(self, padded_im, pad_w, pad_h, original_width, original_height):
        width, height = padded_im.size
        
        left = pad_w
        top = pad_h
        right = width - pad_w
        bottom = height - pad_h
        
        cropped_im = padded_im.crop((left, top, right, bottom))

        resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

        return resized_im

    def _resize_image(self, img, target_size=768):
        width, height = img.size
        
        if width < height:
            scale = target_size / width
        else:
            scale = target_size / height
        
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_img
