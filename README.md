# Pneumonia-Detection
# Pneumonia Detection in Chest X-Rays Using Custom CNN & Transfer Learning

[![Python 3.10+](https://shields.io)](https://python.org)
[![TensorFlow 2.15+](https://shields.io)](https://github.com)
[![Kaggle Dataset](https://shields.io)](https://kaggle.com)

An end-to-end Deep Learning pipeline built with TensorFlow/Keras to automate the detection of Pneumonia from medical chest X-ray images. This project addresses severe class imbalances using custom data augmentation workflows and evaluates performance using custom CNNs alongside Transfer Learning models (ResNet50).

---

## 📌 Project Overview
Pneumonia is a critical lung infection requiring fast and accurate diagnosis. This repository hosts a robust medical imaging pipeline that:
*   **Ingests and Preprocesses** digital X-ray scans.
*   **Mitigates Class Imbalance** via strategic real-time data augmentations (zooms, horizontal flips, shears).
*   **Classifies Images** into normal or infected categories using deep architectures.
*   **Provides Clinical Metrics** by prioritizing High Recall to ensure low false-negative rates in patient screening.

---

## 📂 Repository Structure

```text
├── data/
│   ├── train/            # Balanced and augmented training splits
│   ├── val/              # Validation validation images
│   └── test/             # Untouched baseline evaluation splits
├── models/
│   ├── custom_cnn.h5     # Serialized best-performing custom network weight file
│   └── resnet50_tl.h5    # Final serialized Transfer Learning model
├── notebooks/
│   └── exploratory.ipynb # Initial EDA and dataset distribution mapping
├── scripts/
│   ├── data_loader.py    # Directory parsing, formatting, and augmentation generator
│   └── train.py          # Network compiling, callback logic, and orchestration
├── requirements.txt      # Production system environments manifest
└── README.md             # Project documentation and guide
```

---

## 📊 Dataset Verification
This project leverages the verified, peer-reviewed open dataset hosted on Kaggle:
*   **Source:** [Chest X-Ray Images (Pneumonia) Dataset](https://kaggle.com)
*   **Total Images:** 5,856 JPEG images across 2 categories (Pneumonia / Normal).
*   **Structure Expected:**
    ```text
    chest_xray/
    ├── train/ (NORMAL, PNEUMONIA)
    ├── val/   (NORMAL, PNEUMONIA)
    └── test/  (NORMAL, PNEUMONIA)
    ```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com
   cd pneumonia-detection-cnn
   ```

2. **Configure Environment and Libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Fetch the Raw Dataset**
   Download the source directory directly from Kaggle and place the raw images into the local `data/` folder following the structural layout blueprint.

---

## 🚀 Execution Guide

To initiate the training pipeline with full optimization workflows, execute the automation script:

```bash
python scripts/train.py --epochs 25 --batch_size 32 --learning_rate 0.001
```

---

## 🏗️ Model Architecture Details

### 1. Custom Sequential CNN
Designed to balance memory footprints with robust pattern feature extraction:
*   **Input Layer:** `(150, 150, 3)` dimensional tensor mapping.
*   **Convolution Blocks:** 4 alternating series of 2D Convolution layers paired alongside spatial down-sampling Max Pooling sequences.
*   **Regularization:** Injected `BatchNormalization` and `Dropout` instances (0.2 - 0.3) following each dense dense layers to prevent early overfitting.

### 2. Transfer Learning (ResNet50)
*   Initialized with fixed ImageNet weights.
*   The top dense classification stack was fully removed and swapped out with a global average 2D spatial pooling layer followed by a target single node Sigmoid activation unit.

---

## 📈 Evaluation Matrix Summary

Because this is a critical medical task, success is evaluated using strict clinical metrics over raw accuracy scores:

*   **Confusion Matrix:** Monitored to confirm minimized False Negative allocations.
*   **Recall (Sensitivity):** Optimized to ensure that actual positive pneumonia matches are not missed during screenings.
*   **ROC-AUC Curves:** Assesses class separation capacity across various threshold levels.

---

## 🤝 Contributing
Contributions, feature enhancements, and validation feedback are welcome. Feel free to open an Issue or submit a Pull Request.

## 📝 License
This project is open-source software licensed under the MIT License.
