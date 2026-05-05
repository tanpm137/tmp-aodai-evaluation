import os
import sys
import torch
from pathlib import Path

from PIL import Image
from torchvision import transforms

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
vendor_dir = os.path.join(root_dir, "vendors", "imagedressing")

if vendor_dir not in sys.path:
    sys.path.insert(0, vendor_dir)

from vendors.imagedressing.source.dressing_sd.pipelines.IMAGDressing_v1_pipeline import IMAGDressing_v1
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

from vendors.imagedressing.source.adapter.attention_processor import CacheAttnProcessor2_0, RefSAttnProcessor2_0, CAttnProcessor2_0
from vendors.imagedressing.source.adapter.resampler import Resampler

from source.benchmark.dataset.TMPDataset import TMPDataset

class IMAGDressingVirualDressing:
    def __init__(self, model_ckpt: str = "ckpt/IMAGDressing-v1_512.pt", device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.pipe = self._prepare(model_ckpt, device)
        self.clip_image_processor = CLIPImageProcessor()

    def _prepare(self, model_ckpt, device):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16, device=device)
        tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder").to(dtype=torch.float16, device=device)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder").to(dtype=torch.float16, device=device)
        unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(dtype=torch.float16, device=device)

        # load ipa weight
        image_proj = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
        image_proj = image_proj.to(dtype=torch.float16, device=device)

        # set attention processor
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            else:
                attn_procs[name] = CAttnProcessor2_0(name, hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules = adapter_modules.to(dtype=torch.float16, device=device)

        ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(dtype=torch.float16, device=device)
        ref_unet.set_attn_processor(
            {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()}
        )  # set cache

        # weights load
        model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

        ref_unet_dict = {}
        unet_dict = {}
        image_proj_dict = {}
        adapter_modules_dict = {}
        for k in model_sd.keys():
            if k.startswith("ref_unet"):
                ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
            elif k.startswith("unet"):
                unet_dict[k.replace("unet.", "")] = model_sd[k]
            elif k.startswith("proj"):
                image_proj_dict[k.replace("proj.", "")] = model_sd[k]
            elif k.startswith("adapter_modules"):
                adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
            else:
                pass

        ref_unet.load_state_dict(ref_unet_dict)
        image_proj.load_state_dict(image_proj_dict)
        adapter_modules.load_state_dict(adapter_modules_dict)

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        pipe = IMAGDressing_v1(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                             text_encoder=text_encoder, image_encoder=image_encoder,
                             ImgProj=image_proj,
                             scheduler=noise_scheduler,
                             safety_checker=StableDiffusionSafetyChecker,
                             feature_extractor=CLIPImageProcessor)
        return pipe

    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        guidance_scale: float = 7.5,
        image_scale: float = 1.0,
        num_inference_steps: int = 30,
        size: tuple[int, int] = (576, 768),
        seed: int = 42):
        
        width, height = size
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        img_transform = transforms.Compose([
            transforms.Resize([height, width], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        print(f"Processing {len(dataset)} samples...")
        for sample in dataset:
            filename = sample["filename"]
            print(f"Processing {filename}...")

            text_prompt = "The photo of a person wearing a dress with pose description: " + sample["pose_description"]
            cloth_path = sample["garment_path"]

            prompt = text_prompt + ', best quality, high quality'
            null_prompt = ''
            negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'

            clothes_img = Image.open(cloth_path).convert("RGB")
            clothes_img = self._resize_img(clothes_img)
            vae_clothes = img_transform(clothes_img).unsqueeze(0)
            ref_clip_image = self.clip_image_processor(images=clothes_img, return_tensors="pt").pixel_values

            output = self.pipe(
                ref_image=vae_clothes,
                prompt=prompt,
                ref_clip_image=ref_clip_image,
                null_prompt=null_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                image_scale=image_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            ).images
            
            generated_image = output[0]
            generated_image_name = filename + ".jpg"
            full_output_path = os.path.join(output_dir, generated_image_name)
            generated_image.save(full_output_path)
            
            print(f"Saved to {full_output_path}")

    def _resize_img(self,input_image, max_side=1000, min_side=512, mode=Image.BILINEAR, base_pixel_number=64):
        w, h = input_image.size
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        return input_image