import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class BraTSDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        n = len(subjects)
        if split == 'train':
            self.subjects = subjects[:int(0.8 * n)]
        elif split == 'val':
            self.subjects = subjects[int(0.8 * n):int(0.9 * n)]
        else:
            self.subjects = subjects[int(0.9 * n):]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        path = os.path.join(self.root_dir, subj)
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        for mod in modalities:
            img_path = os.path.join(path, f"{mod}.nii.gz")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing {img_path}")
            img = nib.load(img_path).get_fdata().astype(np.float32)
            images.append(img)

        seg_path = os.path.join(path, "seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int64)

        # Center crop to 128x128x128
        cx, cy, cz = np.array(images[0].shape) // 2
        r = 64
        images = [img[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] for img in images]
        seg = seg[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r]

        # Z-score normalization
        images = [(img - img.mean()) / (img.std() + 1e-8) for img in images]
        images = np.stack(images, axis=0)
        return torch.tensor(images), torch.tensor(seg)