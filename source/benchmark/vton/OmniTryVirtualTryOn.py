import torch
import diffusers
import transformers
import copy
import random
import numpy as np
import torchvision.transforms as T
import math
import peft
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from PIL import Image
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from diffusers import PipelineQuantizationConfig
from vendors.omni_try.source.omnitry.models.transformer_flux import FluxTransformer2DModel
from vendors.omni_try.source.omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline
from source.benchmark.dataset.TMPDataset import TMPDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()


def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            torch_result_dtype = result.dtype
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))


class OmniTryVirtualTryOn:

    def __init__(
        self,
        model_root: str,
        lora_path: str,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        offload: bool = False,
        mixed_precision: str = "bf16",
        should_quantize: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        self.device_str = "cuda" if device.type == "cuda" else "cpu"
        
        match mixed_precision:
            case "fp16":
                self.torch_dtype = torch.float16
            case "bf16":
                self.torch_dtype = torch.bfloat16
            case _:
                self.torch_dtype = torch.float32

        if should_quantize:
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit", 
                quant_kwargs={                  
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True
                }
            )
        else:
            quant_config = None

        if not os.path.exists(model_root):
            raise ValueError("Model root not exists!")

        # init model & pipeline
        transformer = FluxTransformer2DModel.from_pretrained(
            f'{model_root}/transformer'
        ).requires_grad_(False).to(dtype=self.torch_dtype)
        
        self.pipeline = FluxFillPipeline.from_pretrained(
            model_root, 
            transformer=transformer.eval(), 
            torch_dtype=self.torch_dtype,
            quantization_config=quant_config
        )

        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.vae.enable_tiling()
        else:
            self.pipeline.to(self.device_str)

        # insert LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                'x_embedder',
                'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
                'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
                'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
                'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
            ]
        )
        self.pipeline.transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
        self.pipeline.transformer.add_adapter(lora_config, adapter_name='garment_lora')

        with safe_open(lora_path, framework="pt") as f:
            lora_weights = {k: f.get_tensor(k) for k in f.keys()}
            self.pipeline.transformer.load_state_dict(lora_weights, strict=False)

        # hack lora forward
        for n, m in self.pipeline.transformer.named_modules():
            if isinstance(m, peft.tuners.lora.layer.Linear) or (hasattr(m, 'lora_A') and hasattr(m, 'lora_B')):
                m.forward = create_hacked_forward(m)

    def seed_everything(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _generate(self, person_image, object_image, object_class, steps=20, guidance_scale=30.0, seed=-1):
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        self.seed_everything(seed)

        # resize model
        max_area = 1024 * 1024
        oW = person_image.width
        oH = person_image.height

        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
        transform = T.Compose([
            T.Resize((tH, tW)),
            T.ToTensor(),
        ])
        person_image = transform(person_image)

        # resize and padding garment
        ratio = min(tW / object_image.width, tH / object_image.height)
        transform = T.Compose([
            T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
            T.ToTensor(),
        ])
        object_image_padded = torch.ones_like(person_image)
        object_image = transform(object_image)
        new_h, new_w = object_image.shape[1], object_image.shape[2]
        min_x = (tW - new_w) // 2
        min_y = (tH - new_h) // 2
        object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_image

        # prepare prompts & conditions
        object_map = {
            "top clothes": "replacing the top cloth",
            "bottom clothes": "replacing the bottom cloth",
            "dress": "replacing the dress",
            "shoe": "replacing the shoe",

            "earrings": "trying on earrings",
            "bracelet": "trying on bracelet",
            "necklace": "trying on necklace",
            "ring": "trying on ring",

            "sunglasses": "trying on sunglasses",
            "glasses": "trying on glasses",
            "belt": "trying on belt",
            "bag": "trying on bag",
            "hat": "trying on hat",
            "tie": "trying on tie",
            "bow tie": "trying on bow tie",
        }

        prompts = [object_map.get(object_class, "replacing the top cloth")] * 2
        img_cond = torch.stack([person_image, object_image_padded]).to(dtype=self.torch_dtype, device=self.device_str) 
        mask = torch.zeros_like(img_cond).to(img_cond)

        with torch.no_grad():
            img = self.pipeline(
                prompt=prompts,
                height=tH,
                width=tW,    
                img_cond=img_cond,
                mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(self.device_str).manual_seed(seed),
            ).images[0]

        return img

    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        n_steps: int = 20,
        guidance_scale: float = 30.0,
        seed: int = 42,
        size: tuple[int, int] = (768, 1024)
    ):
        width, height = size
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Processing {len(dataset)} samples...")
        for sample in dataset:
            filename = sample["filename"]
            type_str = sample["type"]
            
            vton_img_path = str(sample["human_image"])
            garm_img_path = str(sample["vton_garment_path"])
            
            print(f"Processing {filename}...")
            
            object_type = "dress" if type_str == "female" else "top clothes"
            
            person_image = Image.open(vton_img_path).convert('RGB').resize((width, height))
            garment_image = Image.open(garm_img_path).convert('RGB')
            
            result_image = self._generate(
                person_image, 
                garment_image, 
                object_type, 
                steps=n_steps, 
                guidance_scale=guidance_scale, 
                seed=seed
            )
            
            generated_image_name = filename + ".jpg"
            full_output_path = os.path.join(output_dir, generated_image_name)
            
            result_image.save(full_output_path)
            print(f"Saved to {full_output_path}")
