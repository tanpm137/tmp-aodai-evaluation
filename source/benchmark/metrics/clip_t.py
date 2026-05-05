import os
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms

from source.benchmark.dataset.TMPDataset import TMPDataset

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class InferenceImageTextDataset(Dataset):
    def __init__(self, dataset: TMPDataset, inference_result_path: Path, transform):
        self.dataset = dataset
        self.transform = transform
        
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        self.available_images = {p.stem: p for p in inference_result_path.iterdir() if p.suffix.lower() in valid_extensions}
        
        self.valid_samples = []
        for sample in self.dataset:
            filename = sample["filename"]
            if filename in self.available_images:
                self.valid_samples.append({
                    "image_path": self.available_images[filename],
                    "pose_desc": sample["pose_description"]
                })
            else:
                print(f"Not found image name {filename}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        
        pose_desc = str(sample["pose_desc"])
            
        return image_tensor, pose_desc

def compute_clip_t(dataset: TMPDataset, inference_result_path: Path, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32) -> float:
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
    metric = CLIPScore(
        model_name_or_path="openai/clip-vit-large-patch14"
    ).to(device)
    
    inference_dataset = InferenceImageTextDataset(dataset, inference_result_path, transform)
    
    if len(inference_dataset) == 0:
        return 0.0

    dataloader = DataLoader(
        inference_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=(device == "cuda")
    )
    
    for batch_images, batch_texts in tqdm(dataloader, desc="Calculating CLIP T"):
        metric.update(batch_images.to(device), list(batch_texts))

    final_mean_score = metric.compute()
    return final_mean_score.item() / 100.0
    