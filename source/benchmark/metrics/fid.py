import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics

from source.benchmark.dataset.TMPDataset import TMPDataset

import numpy as np

class FIDKIDImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.mask_paths is not None and idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            if mask_path is not None and Path(mask_path).exists():
                mask = Image.open(mask_path).convert("L").resize(img.size)
                
                img_np = np.array(img)
                mask_np = np.where(np.array(mask) > 128, 1, 0).astype(np.uint8)
                mask_3c = np.stack([mask_np] * 3, axis=-1)
                img = Image.fromarray(img_np * mask_3c)
                
        img_tensor = torch.from_numpy(np.array(img))
        img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor

def compute_fid(dataset: TMPDataset, inference_result_path: Path, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Calculating FID")
    
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    
    real_paths = [s["human_image"] for s in dataset]
    real_masks = [s["human_mask"] for s in dataset]

    generated_mask_dir = inference_result_path / "human_masks"
    
    available_generated_masks = {m.stem: m for m in generated_mask_dir.iterdir() if m.suffix.lower() in valid_extensions} if generated_mask_dir.exists() else {}
    
    # Filter generated images to match only those present in the (possibly filtered) dataset
    valid_filenames = {s["filename"] for s in dataset}
    
    generated_paths = []
    generated_masks = []
    
    for p in inference_result_path.iterdir():
        if p.is_file() and p.suffix.lower() in valid_extensions and p.stem in valid_filenames:
            generated_paths.append(p)
            generated_masks.append(available_generated_masks.get(p.stem))

    if len(real_paths) == 0:
        print("No valid image pairs found for FID calculation.")
        return None

    real_dataset = FIDKIDImageDataset(real_paths, real_masks)
    generated_dataset = FIDKIDImageDataset(generated_paths, generated_masks)
    
    try:
        metrics_dict = calculate_metrics(
            input1=generated_dataset,
            input2=real_dataset,
            cuda=(device == "cuda" or device == "cuda:0"),
            isc=False,
            fid=True,
            kid=False,
            verbose=False
        )
        
        return metrics_dict["frechet_inception_distance"]
    except Exception as e:
        print(f"Error calculating FID/KID: {e}")
        return None