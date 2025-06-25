# YOLO-Multispec

**YOLO-MultiSpec: A Deep Learning Framework for Multispectral Object Detection and Instance Segmentation**

> 📄 This repository contains code associated with the manuscript:
> “YOLO-MultiSpec: A Deep Learning Framework for Multispectral Object Detection and Instance Segmentation”

> 🚧 **Status:** Manuscript currently under review not yet submitted to *Remote Sensing Letters* (RSL).  
> A formal citation and DOI will be added upon acceptance.

---

## 🔍 Overview

**YOLO-MultiSpec** is a deep learning framework for Multispectral Object Detection and Instance Segmentation in high-resolution multi-band imagery.  
It extends YOLOv8/v11/v12 to support additional spectral channels (e.g., RGB + NIR + RedEdge), enabling improved detection performance in spectrally complex environments.

**Key features:**

- ✅ **Multi-band multispectral input support** (e.g., RGB, NIR, RedEdge)
- 🎯 **Enhanced with CBAM and ECA attention modules** for spatial-spectral refinement
- 🚀 **+10% mAP@50 improvement** over standard YOLO RGB using weeds-galore dataset as a case study
- 🧪 **Easy-to-use example case study** – just modify input parameters for custom multispectral datasets

Tested on an annotated agricultural dataset (_Weeds-Galore_), YOLO-MultiSpec outperforms RGB and baseline multispectral YOLO variants in both accuracy and generalization.  
It is designed for rapid integration into geospatial workflows and QGIS toolkits.

---

## 📁 Repository Structure

# NOTE - code being finalised for upload #########################################
```bash
├── datasets/
│   ├── weeds-galore/         # Example multispectral input (optional)
├── utils/
│   └── patch_backbone_with_attention.py
│   └── mod_pt_model.py           # CBAM, ECA definitions
├── models/
├── train_evaluate.py                  # Train/Evaluation yolu script
├── README.md
└── LICENSE
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch >= 1.10
- OpenCV, NumPy, tifffile
- Ultralytics v8+

Install:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

TO BE UPDATED############################
Train/Evaluate:
```bash
python train_evaluate.py --data data.yaml --imgsz 1024 --epochs 100 --device 0
```

---

## 📜 Citation

```txt
This work is currently under review pre-submission.
Please check back for the official citation once accepted.
```

---

## 📬 Contact

For questions or collaborations, contact:
[Andrew Esparon] – [andrew.esparon@cdu.edu.au]  
[www.cdu.edu.au]

---

## 📄 License

See [LICENSE_YOLO_Multispec.txt] for details.
