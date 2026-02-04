# MS-HVT: Multi-Scale Hierarchical Vision Transformer for Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Official implementation of **MS-HVT (Multi-Scale Hierarchical Vision Transformer)** for accurate and reliable **brain tumor segmentation** from multimodal MRI.

This repository implements a **scale-aware hierarchical transformer** with entropy-guided token pruning and adaptive fusion, designed to preserve fine tumor boundaries while modeling global anatomical context.

---

## 🧠 Method Overview

**MS-HVT** introduces:
- **Scale-Conditional Attention (SCA)** for resolution-aware contextual reasoning  
- **Hierarchy-Aware Token Decimation (HATD)** to remove redundant background tokens  
- **Adaptive Gated Fusion (AGF)** for dynamic multi-scale feature integration  
- **Uncertainty-aware learning** for improved calibration and reliability  

---

## 🧩 Architecture Summary

- **Input**: Multimodal MRI (T1, T1ce, T2, FLAIR)
- **Multi-scale patch sizes**: 8×8, 16×16, 32×32
- **Hierarchical transformer encoder**
- **Entropy-guided token pruning**
- **Adaptive multi-scale fusion**
- **Output**: Pixel/voxel-wise brain tumor segmentation

---

## 📁 Repository Structure

```
Multi-Scale-Hierarchical-Vision-Transformer-for-BT-Segmentation/
│
├── Inference/
│
├── models/
│   ├── ms_hvt.py
│   └── layers/
│       ├── __init__.py
│       ├── sca.py
│       ├── hatd.py
│       └── agf.py
│
├── utils/
│   └── losses.py
│
├── datadataset.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## 📁 Datasets
Brain Tumor Image DataSet : Semantic Segmentation 
https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation?select=test

Indk214
https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification 

Brain Tumor Segmentation
A 2D brain tumor segmentation dataset: https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation?select=masks

Brain tumor segmentation 
https://www.kaggle.com/datasets/tinashri/brain-tumor-segmentation-datasets

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train
```bash
python train.py
```

### Inference
```bash
python inference.py
```

---

## 📄 Citation

```bibtex
@article{MSHVT2026,
  title={Multi-Scale Hierarchical Vision Transformer for Brain Tumor Segmentation},
  author={Indrakumar K, Ravikumar M},
  journal={},
  year={2026}
}
```

---

## 📜 License
MIT License.
