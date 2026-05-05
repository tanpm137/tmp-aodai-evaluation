from diffusers import PipelineQuantizationConfig
import os
import sys
from pathlib import Path
import csv

import torch
from PIL import Image
from torchvision import transforms

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
vendor_dir = os.path.join(root_dir, "vendors", "jco_mvton")

if vendor_dir not in sys.path:
    sys.path.insert(0, vendor_dir)

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from vendors.jco_mvton.source.flux.transformer_flux import FluxTransformer2DModel
from source.benchmark.dataset.TMPDataset import TMPDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

class JcoVirtualTryOn:

    def __init__(
        self,
        model_root: str = "black-forest-labs/FLUX.1-dev",
        model_weight_dress: str = None,
        model_weight_upper: str = None,
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

        self.model_id = model_root
        self.extra_branch_num = 2
        self.mode = 2

        self.pipe_dress = None
        self.pipe_upper = None

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

        if model_weight_dress is not None:
            self.pipe_dress = FluxPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=self.torch_dtype,
                transformer=self._make_transformer(model_weight_dress),
                quantization_config=quant_config,
            )
            if offload:
                self.pipe_dress.enable_model_cpu_offload()
            else:
                self.pipe_dress.to(self.device_str)

        if model_weight_upper is not None:
            self.pipe_upper = FluxPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=self.torch_dtype,
                transformer=self._make_transformer(model_weight_upper),
                quantization_config=quant_config,
            )
            if offload:
                self.pipe_upper.enable_model_cpu_offload()
            else:
                self.pipe_upper.to(self.device_str)

    def _make_transformer(self, trained_weight):
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            subfolder="transformer",
            extra_branch_num=self.extra_branch_num,
            low_cpu_mem_usage=False
        ).to(self.device_str)
        
        for j in range(self.extra_branch_num):
            if self.mode == 1:
                transformer.extra_embedder[j].load_state_dict(transformer.x_embedder.state_dict())
            for i in range(transformer.config.num_layers):
                transformer.transformer_blocks[i].attn.extra_to_q[j].load_state_dict(transformer.transformer_blocks[i].attn.to_q.state_dict())
                transformer.transformer_blocks[i].attn.extra_to_k[j].load_state_dict(transformer.transformer_blocks[i].attn.to_k.state_dict())
                transformer.transformer_blocks[i].attn.extra_to_v[j].load_state_dict(transformer.transformer_blocks[i].attn.to_v.state_dict())
                if self.mode == 1:
                    transformer.transformer_blocks[i].extra_norm1[j].load_state_dict(transformer.transformer_blocks[i].norm1.state_dict())
                    transformer.transformer_blocks[i].extra_norm2[j].load_state_dict(transformer.transformer_blocks[i].norm2.state_dict())
                    transformer.transformer_blocks[i].extra_ff[j].load_state_dict(transformer.transformer_blocks[i].ff.state_dict())
                    transformer.transformer_blocks[i].attn.extra_to_out[0][j].load_state_dict(transformer.transformer_blocks[i].attn.to_out[0].state_dict())
                    transformer.transformer_blocks[i].attn.extra_to_out[1][j].load_state_dict(transformer.transformer_blocks[i].attn.to_out[1].state_dict())
                    transformer.transformer_blocks[i].attn.extra_norm_q[j].load_state_dict(transformer.transformer_blocks[i].attn.norm_q.state_dict())
                    transformer.transformer_blocks[i].attn.extra_norm_k[j].load_state_dict(transformer.transformer_blocks[i].attn.norm_k.state_dict())
            if self.mode == 2:
                for i in range(transformer.config.num_single_layers):
                    transformer.single_transformer_blocks[i].attn.extra_to_q[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_q.state_dict())
                    transformer.single_transformer_blocks[i].attn.extra_to_k[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_k.state_dict())
                    transformer.single_transformer_blocks[i].attn.extra_to_v[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_v.state_dict())
        
        state_dict = torch.load(trained_weight, map_location="cpu")
        if 'module' in state_dict:
            state_dict = state_dict['module']
        transformer.load_state_dict(state_dict, strict=False)
        
        return transformer

    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        n_steps: int = 30,
        guidance_scale: float = 3.5,
        seed: int = 42,
        size: tuple[int, int] = (768, 1024)
    ):
        width, height = size
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        transform_person = transforms.Compose([
            transforms.Resize(size=(height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        transform_cloth = transforms.Compose([
            transforms.Resize(size=(height, height)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"Processing {len(dataset)} samples...")
        for sample in dataset:
            filename = sample["filename"]
            type_str = sample["type"]
            
            vton_img_path = str(sample["human_image"])
            garm_img_path = str(sample["vton_garment_path"])
            
            print(f"Processing {filename}...")
            
            is_dress = type_str == "female"
            pipe = self.pipe_dress if is_dress else self.pipe_upper
            
            if pipe is None:
                print(f"Warning: Pipeline for type '{type_str}' (is_dress={is_dress}) is not initialized. Skipping.")
                continue

            person_image = Image.open(vton_img_path).convert("RGB").resize((width, height))
            garment_image = Image.open(garm_img_path).convert("RGB").resize((height, height))
            
            person_tensor = transform_person(person_image)
            cloth_tensor = transform_cloth(garment_image)
            prompt = "A fashion model wearing stylish clothing, detailed textures, realistic lighting, fashion photography style."
            
            with torch.inference_mode():
                result_image = pipe(
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    prompt=prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=guidance_scale,  
                    height=height,
                    width=width,
                    cloth_img=cloth_tensor,
                    person_img=person_tensor,
                    extra_branch_num=self.extra_branch_num,
                    mode=self.mode,
                    max_sequence_length=77,
                ).images[0]
            
            generated_image_name = filename + ".jpg"
            full_output_path = os.path.join(output_dir, generated_image_name)
            
            result_image.save(full_output_path)
            print(f"Saved to {full_output_path}")
