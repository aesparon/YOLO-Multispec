# YOLO-Multispectral 🚀

<!--
  YOLO-Multispectral: Modified Ultralytics YOLOv8/v11 for 4- and 5-band multispectral object detection and segmentation.
  Keywords: YOLO multispectral, multispectral object detection, RGBN, RGB+NIR, YOLOv8, YOLOv11, drone imagery, vegetation segmentation, CBAM, ECA, geospatial deep learning, QGIS, remote sensing.
-->

![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/github/license/aesparon/YOLO-Multispectral)
![Issues](https://img.shields.io/github/issues/aesparon/YOLO-Multispectral)
![Stars](https://img.shields.io/github/stars/aesparon/YOLO-Multispectral)

## 🔍 What is YOLO-Multispectral?

**YOLO-Multispectral** is an open-source implementation of **YOLOv8 and YOLOv11** modified to support 5+ **multispectral remote sensing imagery** (e.g., RGB, NIR, RedEdge) for **object detection** and **semantic segmentation**, with geospatial metadata support and attention-enhanced backbones (CBAM, ECA).

## 🌿 Why Multispectral Object Detection?

Traditional RGB-based models often miss subtle vegetation features. By integrating **NIR** and **RedEdge** bands, this repo improves vegetation segmentation and monitoring in ecological and agricultural applications.

## 💡 Features

- ✅ 4+ band TIFF multispectral input support
- ✅ Apply transfer learning to multispectral imagery for improved accuracy on limited training data
- ✅ SpectralConv, CBAM, ECA, DropBlock, GroupNorm enhancements
- ✅ Mask + Box segmentation via YOLOv8/v11/v12 (NOTE - v12 Currrently only supporting only bounding boxes)
- ✅ Geospatial compatibility for QGIS + tiled workflows

## 🧠 Model Enhancements

| Module     | Purpose                        | Source Reference |
|------------|--------------------------------|------------------|
| CBAM       | Channel + Spatial Attention    | [CBAM 2018](https://arxiv.org/abs/1807.06521) |
| ECA        | Efficient Channel Attention    | [ECA 2020](https://arxiv.org/abs/1910.03151) |
| SpectralConv | Band-specific Filtering     | [Inspired by 3D-HSI CNNs] |
| DropBlock  | Regularization                | [DropBlock 2018](https://arxiv.org/abs/1810.12890) |
| GroupNorm  | Small batch normalization      | [GN 2018](https://arxiv.org/abs/1803.08494) |

## 📦 Dataset Support

- [x] 5/10 band MicaSense/RedEdge drone imagery
- [x] LLVIP RGB-IR visible-infrared dataset
- [x] WeedsGalore RGB+NIR segmentation benchmark

## 🔗 Citation

Please cite this repo if it supports your work:

```bibtex
@misc{yolo-multispectral,
  author = {Esparon, Andrew},
  title = {YOLO-Multispectral: Multispectral Object Detection and Segmentation using YOLOv8/v11},
  year = {2025},
  howpublished = {\url{https://github.com/aesparon/YOLO-Multispectral}},
}
```

---

## 🎥 Watch this for overview (UNDER CONSTUCTION)

[![Coming soon](https://img.shields.io/badge/Demo-YouTube-red)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

### Run Instantly on Google Colab (under construction - completion by 11/7/2015)
<a href="https://colab.research.google.com/github/aesparon/YOLO-MultiSpectral/blob/main/examples/notebooks/YOLO-MultiSpectral_demo.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="height:30px;">
</a>


## 🧠 Author

Developed by **Andrew Esparon**  
Remote Sensing AI Researcher | QGIS + PyTorch Ecosystem  
<!--[GitHub](https://github.com/aesparon) -->
