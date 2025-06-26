# YOLO-MultiSpectral

![PyTorch](https://img.shields.io/badge/PyTorch-≥1.10-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Under%20Review-orange)
![Stars](https://img.shields.io/github/stars/aesparon/YOLO-Multispectral?style=social)

**YOLO-MultiSpectral: A Deep Learning Framework for Multispectral Object Detection and Instance Segmentation**

> **Real-time object detection and segmentation in 4+ channel multispectral imagery, powered by enhanced YOLOv8/YOLOv11+ backbones.**

---

## 🔍 Overview

YOLO-MultiSpectral is a custom deep learning framework designed for object detection and instance segmentation in **multispectral remote sensing imagery**.  
It supports **4+ channel TIFF inputs** (e.g., RGB + NIR + RedEdge and other custom bands) and integrates spatial-spectral attention modules like **CBAM** and **ECA** to improve generalization.

---

## ✨ Key Features

- 🧠 Supports RGB, NIR, RedEdge, and other custom bands
- 🎯 CBAM and ECA attention modules for spectral feature refinement
- 📈 +10% mAP@50 gain vs. baseline RGB-only YOLO
- 🚜 Optimized for precision agriculture, UAV imagery, and QGIS workflows
- 🔁 Easy to integrate, fine-tune, and extend for custom projects

---

## 📁 Repository Structure

```
├── datasets/
│   └── weeds-galore/             # Optional example dataset
├── utils/
│   ├── patch_backbone_with_attention.py
│   └── mod_pt_model.py          # CBAM, ECA definitions
├── models/
├── train_evaluate.py            # Main training/evaluation logic
├── requirements.txt
├── README.md
├── CITATION.cff
└── LICENSE
```

---

## 📦 Installation

### Requirements:
- Python 3.8+ (currently supports 3.10)
- PyTorch ≥ 1.10
- Ultralytics YOLOv8+
- OpenCV, NumPy, tifffile

```bash
pip install -r requirements.txt
```

### Optional (Anaconda Environment):
```bash
conda create -n yolo-multispectral python=3.8
conda activate yolo-multispec
pip install -r requirements.txt
```

---

## 🧪 Quick Start using weed-galore dataset

Train a YOLO-MultiSpectral model on 5-band imagery: 

### Option 1: Run Instantly on Google Colab
- No setup required; 
<sub>Tip: Right-click the badge and choose "Open in new tab" to keep this page open.</sub>
<a href="https://colab.research.google.com/github/aesparon/YOLO-Multispectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="height:50px;">
</a>

---

### Option 2: Run locally
```bash
python train_evaluate.py --data data.yaml --imgsz 1024 --epochs 100 --device 0
```

- Make sure your `.yaml` file reflects the number of input channels and class labels.
- Outputs are saved in the `runs/` folder with metrics, predictions, and checkpoints.

---

## 🗂️ Input Format

- **Image format**: `.tif`, `.png` with 4+ spectral bands (uint8 )  ( current uint16 )
- **Labels**: YOLO format `.txt` files with polygon coordinates for segmentation or bounding boxes
- **Data YAML**: follows Ultralytics `data.yaml` format, with custom paths and class names

---

- Ensure your `.yaml` file reflects the number of input channels and class labels.
- Outputs are saved in the `runs/` folder with metrics, predictions, and checkpoints.



## 🗂️ Input Format

- **Image format**: `.tif` or `.png` with 4+ spectral bands (uint8 or uint16)
- **Labels**: YOLO format `.txt` files with polygon coordinates for segmentation or bounding boxes
- **Data YAML**: follows Ultralytics `data.yaml` format, with custom paths and class names

---

## 🧩 Extending YOLO-MultiSpec

YOLO-MultiSpectral is modular by design. You can:
- 💡 Add new attention modules (e.g., SE, CBNet, Transformer blocks)
- 🔄 Swap backbone (e.g., ResNet, CSPDarknet, ConvNeXt)
- 🧠 Modify loss functions (e.g., Focal-EIoU, GIoU)
- 📊 Integrate NDVI/NDRE indices into the model input

We welcome contributions — see [CONTRIBUTING.md](CONTRIBUTING.md)!

---

## 📈 Case Study: Weeds-Galore Dataset

This repo includes training support for the **Weeds-Galore** dataset:
- 5-band UAV imagery (R, G, B, NIR, RedEdge)
- Classes: *maize, amaranth, grass, quickweed, other*
- Achieved **+10% mAP@50** over YOLOv8 (RGB)

---

## 📜 Citation

```txt
This work is currently under review.
Please check back for the formal citation once published.
```

---

## 📬 Contact

**Andrew Esparon**  
📧 andrew.esparon@cdu.edu.au  
🌐 [Charles Darwin University](https://www.cdu.edu.au)

---

## 📝 License

MIT License – See [LICENSE_YOLO_Multispectral.txt](LICENSE_YOLO_Multispec.txt)

---

## 🔖 Keywords

`Multispectral` · `YOLOv8` · `YOLOv11` · `YOLOv12` · `deep learning ` · `CNN`  · `Object Detection` · `Instance Segmentation` · `Remote Sensing AI` · `Agricultural Deep Learning` · `CBAM` · `5-channel Imagery` · `Precision Agriculture` · `Geospatial Deep Learning`