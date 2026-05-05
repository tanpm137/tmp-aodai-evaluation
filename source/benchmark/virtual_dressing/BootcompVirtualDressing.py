import os
import torch
import sys

from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
vendor_dir = os.path.join(root_dir, "vendors", "bootcomp")

if vendor_dir not in sys.path:
    sys.path.insert(0, vendor_dir)

from vendors.bootcomp.source.compose_pipeline_xl import StableDiffusionXLPipeline as ComposePipeline
from vendors.bootcomp.source.unet_hacked_tryon import UNet2DConditionModel
from vendors.bootcomp.source.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_encoder

from source.benchmark.dataset.TMPDataset import TMPDataset

class BootcompVirtualDressing:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "SG161222/RealVisXL_V3.0",
        unet_encoder_ckpt: str = "omniousai/BootComp",
        pretrained_vae_path: str = "madebyollin/sdxl-vae-fp16-fix",
        mixed_precision: str = "fp16",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):

        self.device = device

        match mixed_precision:
            case "fp16":
                weight_dtype = torch.float16
            case "bf16":
                weight_dtype = torch.bfloat16
            case _:
                weight_dtype = torch.float32

        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",rescale_betas_zero_snr=True)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")

        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
        text_encoder_2.requires_grad_(False)
        text_encoder_2.to(device, dtype=weight_dtype)

        vae = AutoencoderKL.from_pretrained(pretrained_vae_path)
        vae.requires_grad_(False)
        vae.to(device,dtype=weight_dtype)

        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        unet.requires_grad_(False)
        unet.to(device, dtype=weight_dtype)

        unet_encoder = UNet2DConditionModel_encoder.from_pretrained(unet_encoder_ckpt, subfolder="comp", use_safetensors=False)
        unet_encoder.requires_grad_(False)
        unet_encoder.to(device, dtype=weight_dtype)

        self.newpipe = ComposePipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=unet,
            vae= vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=noise_scheduler,
            torch_dtype=torch.float16,
            add_watermarker=False
        ).to(device)
        self.newpipe.unet_encoder = unet_encoder
        self.weight_dtype = weight_dtype
    
    def process(
        self,
        dataset: TMPDataset,
        output_dir: str,
        guidance_scale: float = 4.0,
        cloth_scale: float = 2.0,
        num_inference_steps: int = 30,
        size: tuple[int, int] = (576, 768),
        seed: int = 42):

        width, height = size
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        with torch.amp.autocast('cuda'), torch.no_grad():
            print(f"Processing {len(dataset)} samples...")
            for sample in dataset:
                filename = sample["filename"]
                print(f"Processing {filename}...")

                full_prompt = "The photo of a person wearing a dress with pose description: " + sample["pose_description"]
                cloth_desc = "A photo of dress"

                with torch.inference_mode():
                    (prompt_embeds_garment, _, pooled_prompt_embeds_garment, _) = self.newpipe.encode_prompt(cloth_desc, num_images_per_prompt=1, do_classifier_free_guidance=False)
                
                cloth_image = self._transformed_image(sample["garment_path"], width, height).to(self.device, dtype=self.weight_dtype)
                cloth_image = cloth_image.unsqueeze(0)
                
                generated_image = self.newpipe(
                    prompt=full_prompt,
                    num_inference_steps=num_inference_steps,
                    img_mat=cloth_image,
                    prompt_ref=prompt_embeds_garment.to(self.device, dtype=self.weight_dtype),
                    pooled_prompt_embeds_ref=pooled_prompt_embeds_garment.to(self.device, dtype=self.weight_dtype),
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    cloth_scale=cloth_scale,
                    generator=generator,
                )[0][0]

                generated_image_name = filename + ".jpg"
                generated_path = Path(output_dir) / generated_image_name

                generated_image.save(generated_path)
                print(f"Saved to {generated_path}")
    
    def _transformed_image(self, image_path: str, width: int, height: int) -> torch.Tensor:
        cloth_image = Image.open(image_path).resize((width, height))
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        cloth_image = transform(cloth_image)
        return cloth_image