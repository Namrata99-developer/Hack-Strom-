# 🚙 Off-Road Semantic Segmentation: Optimized Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains a robust, end-to-end training and inference pipeline for off-road semantic segmentation. It serves as a stronger replacement for standard starter scripts, resolving critical data and architectural bugs to maximize performance.

## ✨ Core Improvements over Baseline

Our pipeline addresses several critical issues found in the provided starter code:
* **Correct Class Handling:** Uses the correct 10 classes defined in the dataset documentation, specifically retaining the `Flowers (600)` class which was previously dropped by label bugs.
* **Preserved Mask Integrity:** Implements safe resizing logic to avoid corrupting segmentation masks during preprocessing.
* **Full Architecture Training:** Trains the complete segmentation model end-to-end, rather than isolating and training only a small prediction head.
* **Accurate Output Formatting:** Saves proper validation and test predictions in raw label IDs for exact evaluation matching.

---

## 📊 Our Updated Code Results

> **VALIDATION IOU SCORE:** 0.6905
> **TEST IOU SCORE:** 0.3108

> **MODEL USED:** DeeplabV3+ with Restnet101 and Imagenet(Pretrained Model) 
---

---

## 📊 Dataset & Metrics Insights

> **Important Metric Note:** `0.80` mean IoU is *not* the same thing as `80%` pixel accuracy.
> * **IoU** (Intersection over Union) represents class overlap: `TP / (TP + FP + FN)`.
> * **mIoU** averages classes more fairly than raw pixel accuracy, ensuring that rare classes heavily impact the score. For example, sky and landscape might look perfect, while flowers or logs fail completely.

### Real Data Analysis
* **Initial Baseline:** The starter baseline provided by Duality achieved `0.2478` mIoU.
* **Class Imbalance:** `Logs` are extremely rare in both the train and validation splits.
* **Test Split Variance:** The local test split contains only a subset of the total classes, meaning local test IoU and hidden-test IoU may differ significantly.

---

## 📂 Directory Structure

Point `dataset.root_dir` in your configuration file to the main folder containing the dataset. The expected structure is:

```text
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/
    └── Segmentation/
