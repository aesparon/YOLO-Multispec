# YOLO-MultiSpectral: Multispectral Object Detection and Segmentation with YOLO

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Under%20Review-orange)]()
[![Stars](https://img.shields.io/github/stars/aesparon/YOLO-MultiSpectral.svg?style=social)](https://github.com/aesparon/YOLO-MultiSpectral/stargazers)
[![DOI](https://zenodo.org/badge/123456789.svg)](https://zenodo.org/badge/latestdoi/123456789)

> **Object detection and segmentation in 4+ channel multispectral imagery, powered by enhanced YOLOv3-v12+ backbones.**

---

## 🔍 Overview

**YOLO-MultiSpectral** is a deep learning framework based YOLO that supports multispectral imagery with 4 or more bands (e.g., RGB + NIR + RedEdge) for object detection and instance segmentation. Designed for UAV imagery, precision agriculture, and environmental monitoring, it integrates spatial-spectral attention mechanisms like **CBAM** and **ECA** to enhance accuracy and generalization.

---

## 🚀 Features
- ✅ 4+ band multispectral TIFF input support (current support for uint8 and soon to be modified for uint16)
- ✅ CBAM and ECA attention modules for spectral feature enhancement
- ✅ +10% mAP@50 gain over standard RGB YOLO
- ✅ Optimized for QGIS workflows and UAV imagery
- ✅ Modular architecture for easy integration and extension

---

## 🧠 Model Architecture
- Backbone: Modified YOLO (v3-v12) with CBAM, ECA, and spectral convolutions
- Input :  images (png,tiff required for 5 channel support)
- Output: Bounding boxes + instance masks


## 🧪 Quick Start using weed-galore dataset

### Option 1: Run Instantly on Google Colab
<a href="https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="height:50px;">
</a>

## 🎥 Link to youtube demo usage

[![Coming soon](https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)


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

# requirements txt -> refer colab example
<!-- - Python 3.8+
- PyTorch ≥ 1.10  # recomended version XXX
- Ultralytics YOLO
- OpenCV, NumPy, tifffile, PyYAML -->

```bash
pip install -r requirements.txt
```

### Optional (Anaconda Environment):
```bash
conda create -n yolo-multispectral python=3.10
conda activate yolo-multispectral
pip install -r requirements.txt
```

---

## 🧪 Quick Start using weed-galore dataset

### Option 1: Run Instantly on Google Colab
<a href="https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="height:50px;">
</a>

### Option 2: Run locally
```bash
python train_evaluate.py --data data.yaml --imgsz 1024 --epochs 100 --device 0
```
- Ensure your `.yaml` file reflects the number of input channels and class labels.
- Outputs are saved in the `runs/` folder with metrics, predictions, and checkpoints.

---

## 🗂️ Input Format
- **Image format**: `.tif`, `.png` with 4+ spectral bands (current support for uint8 and soon to be modified for uint16)
- **Labels**: YOLO format `.txt` files with polygon coordinates or bounding boxes
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
- Achieved **+10% mAP@50** over YOLO (RGB)

---

## 📊 Evaluation
Run validation on a 5-band TIFF dataset:
```bash
from ultralytics import YOLO
model = YOLO('yolov8n-seg.yaml')
model.val(data='data.yaml')
```

---

## 📦 Citation
If you use this project in your research:
```bibtex
@misc{yolo-multispectral,
  author       = {Andrew Esparon},
  title        = {YOLO-MultiSpectral},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/aesparon/YOLO-MultiSpectral}},
  doi          = {10.5281/zenodo.123456789}
}
```

---

## 📬 Contact
**Andrew Esparon**  
📧 andrew.esparon@cdu.edu.au  
🌐 [Charles Darwin University](https://www.cdu.edu.au)

---

## 📝 License
AGPL-3.0 License – See [LICENSE](LICENSE)

---

## 🔖 Keywords
`Multispectral` · `YOLOv8` · `YOLOv11` · `YOLOv12` · `deep learning` · `weeds galore` · `CNN` · `Object Detection` · `Instance Segmentation` · `Remote Sensing AI` · `Agricultural Deep Learning` · `CBAM` · `5-channel Imagery` · `Precision Agriculture` · `Geospatial Deep Learning`
