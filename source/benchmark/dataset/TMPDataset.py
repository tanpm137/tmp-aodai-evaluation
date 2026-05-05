from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from enum import Enum

class TMPDataset(Dataset):

    class Phase(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self, dataset_path_str: str, phase: Phase, frontal_only: bool = False):
        super(TMPDataset, self).__init__()
        self._items = self._load_items(dataset_path_str, phase, frontal_only)

    def _load_items(self, dataset_path_str: str, phase: Phase, frontal_only: bool = False):

        dataset_path = Path(dataset_path_str)
        pose_desc_path = dataset_path / "pose_desc.csv"
        image_data_path = dataset_path / phase.value

        cloth_images_path = image_data_path / "cloths"
        cloth_masks_path = image_data_path / "cloth_masks"
        human_images_path = image_data_path / "images"
        human_masks_path = image_data_path / "human_masks"

        ref_images_names = sorted([f.stem for f in human_images_path.iterdir() if f.is_file()])
        cloth_masks = sorted([f for f in cloth_masks_path.iterdir() if f.is_file()])
        human_masks = sorted([f for f in human_masks_path.iterdir() if f.is_file()])
        human_images = sorted([f for f in human_images_path.iterdir() if f.is_file()])

        ref_images_path = zip(ref_images_names, cloth_masks, human_masks, human_images)

        pose_desc_df = pd.read_csv(pose_desc_path, index_col='filename')
        ref_images_df = pd.DataFrame(ref_images_path, columns=['filename', 'cloth_mask', 'human_mask', 'human_image'])

        info_df = pd.merge(pose_desc_df, ref_images_df, on='filename', how='right')

        cloth_df = pd.DataFrame({
            'garment_name': [f.stem for f in cloth_images_path.iterdir() if f.is_file()],
            'garment_path': cloth_images_path.iterdir()
        })
        info_df['garment_name'] = info_df['filename'].apply(lambda x: x.split('_')[0])
        final_df = pd.merge(info_df, cloth_df, on='garment_name', how='inner')

        vton_pair_path = Path(image_data_path) / f"{phase.value}_pair.csv"
        vton_pair_df = pd.read_csv(vton_pair_path, dtype=str)
        
        final_df = pd.merge(final_df, vton_pair_df, left_on='filename', right_on='human', how='inner')
        
        cloth_stem_to_path = {f.stem: f for f in cloth_images_path.iterdir() if f.is_file()}
        human_stem_to_path = {f.stem: f for f in human_images_path.iterdir() if f.is_file()}
        
        final_df['vton_garment_path'] = final_df['garment'].apply(lambda x: cloth_stem_to_path.get(str(x)))
        final_df['vton_compared_path'] = final_df['compared'].apply(lambda x: human_stem_to_path.get(str(x)))

        if frontal_only:
            final_df = final_df[final_df['pose'] == 'frontal_pose']
        
        return final_df.to_dict('records')

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


    


        