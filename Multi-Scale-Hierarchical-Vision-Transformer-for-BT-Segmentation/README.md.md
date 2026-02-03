# MS-HVT: Multi-Scale Hierarchical Vision Transformer for Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This repository implements the **MS-HVT** architecture for multi-institutional brain tumor segmentation from multi-sequence MRI, featuring:
- **Scale-Conditional Attention (SCA)**
- **Hierarchy-Aware Token Decimation (HATD)**
- **Adaptive Gated Fusion (AGF)**

Designed for the **BraTS 2021** challenge, it enables scale-adaptive contextual reasoning while preserving tumor boundary details.

## 🧠 Architecture Overview
- Input: 4-channel 3D MRI (T1, T1ce, T2, FLAIR)
- Multi-scale patch embedding at 8×8×8, 16×16×16, 32×32×32
- Hierarchical encoder with intra- and inter-scale attention
- Entropy-guided token pruning (HATD)
- Learnable multi-scale fusion (AGF)
- Output: Voxel-wise segmentation of ET, TC, WT

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt