# Deepfake Detection using CNN + LSTM

## Project Overview

This project implements a **Deepfake Detection System** using a **Convolutional Neural Network (CNN)** combined with a **Long Short-Term Memory (LSTM)** network. The model is trained on the **FaceForensics++ (C23)** dataset to classify videos as **Real** or **Fake**. CNN extracts spatial features from individual video frames, while LSTM captures temporal dependencies across consecutive frames, enabling effective detection of manipulated facial videos.

---

## Features

* Deepfake detection using a CNN + LSTM architecture.
* Automatic extraction of frames from video files.
* Binary classification of videos as **Real** or **Fake**.
* Training with TensorFlow/Keras.
* Memory-efficient video data generator for large datasets.
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC-AUC Score
  * Confusion Matrix
* Visualization of:

  * Training vs Validation Accuracy
  * Training vs Validation Loss
  * ROC Curve
  * Confusion Matrix
* Model saving and prediction on unseen videos.

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib
* Google Colab
* FaceForensics++ (C23) Dataset

---

## Folder Structure

```text
Deepfake-Detection-CNN-LSTM/
│
├── notebook/
│   └── Deepfake_Detection.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── model.py
│   └── utils.py
│
├── dataset/
│   └── README.md
│
├── results/
│   ├── classification_report.txt
│   ├── metrics_summary.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_validation_accuracy.png
│   └── training_validation_loss.png
│
├── saved_model/
│   └── deepfake_detector.keras
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Dataset

This project uses the **FaceForensics++ (C23)** dataset.

**Dataset Link:**
https://www.kaggle.com/datasets/xdxd003/ff-c23

The dataset contains:

* 7,000 videos
* 1,000 real videos
* 6,000 manipulated videos
* Multiple deepfake generation methods:

  * DeepFakeDetection
  * Deepfakes
  * Face2Face
  * FaceShifter
  * FaceSwap
  * NeuralTextures

Due to storage limitations, the dataset is **not included** in this repository. Please download it separately using the link above and follow the instructions in `dataset/README.md`.

---

## Installation

1. Clone this repository.

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the FaceForensics++ (C23) dataset.

4. Extract the dataset.

5. Update the dataset path in the training script.

6. Run the training notebook or `train.py` to train the model.

---

## Results

The model generates the following evaluation outputs:

* Classification Report
* Confusion Matrix
* ROC Curve
* Training vs Validation Accuracy Graph
* Training vs Validation Loss Graph
* Metrics Summary

All generated outputs are available in the `results/` directory.

---

## Future Improvements

* Improve model performance using transfer learning (e.g., EfficientNet or MobileNet).
* Integrate face detection before frame extraction.
* Support multi-class classification for different deepfake generation techniques.
* Develop a real-time deepfake detection web application using Flask or Streamlit.
* Optimize the model for faster inference on edge devices.
* Explore Vision Transformers (ViTs) and attention-based architectures for improved accuracy.
