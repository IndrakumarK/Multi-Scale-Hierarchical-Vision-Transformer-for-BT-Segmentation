import argparse
import os
import torch
import nibabel as nib
import numpy as np
from models import MS_HVT

def infer(subject_path, model_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MS_HVT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    modalities = ['t1', 't1ce', 't2', 'flair']
    images = []
    for mod in modalities:
        img = nib.load(os.path.join(subject_path, f"{mod}.nii.gz")).get_fdata().astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)
        cx, cy, cz = np.array(img.shape) // 2
        r = 64
        img = img[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r]
        images.append(img)
    images = np.stack(images, axis=0)[None, ...]  # [1, 4, 128, 128, 128]

    with torch.no_grad():
        pred = model(torch.tensor(images, dtype=torch.float32).to(device))
        seg = pred.argmax(dim=1).cpu().squeeze().numpy()

    affine = nib.load(os.path.join(subject_path, "t1.nii.gz")).affine
    seg_img = nib.Nifti1Image(seg.astype(np.uint8), affine)
    os.makedirs(output_dir, exist_ok=True)
    nib.save(seg_img, os.path.join(output_dir, "prediction.nii.gz"))
    print(f"Prediction saved to {output_dir}/prediction.nii.gz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default='./results')
    args = parser.parse_args()
    infer(args.input_dir, args.checkpoint, args.output_dir)