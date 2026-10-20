# Industrial Surface Crack Detection using CNN

A deep learning project that classifies surface images as **Crack** or **No Crack** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It covers the full pipeline: dataset splitting, augmentation, training with callbacks, evaluation, and single-image prediction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Features
- Automatic 70% / 15% / 15% train/validation/test split
- Data augmentation (rotation, zoom, shifts, horizontal flip)
- 4-block CNN with Batch Normalization and Dropout
- Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- Accuracy and loss curves, confusion matrix, classification report
- Single-image prediction function
- Reproducible runs with fixed random seeds

## Dataset
The dataset is not included in this repository. Place it in the project root in this structure:

```
CrackDataset/
├── Positive/   # images with cracks
└── Negative/   # images without cracks
```

The script automatically creates `Processed_CrackDataset/` with `train`, `validation` and `test` folders, each containing `Crack` and `NoCrack` subfolders.

> Keras assigns class indices alphabetically (`Crack = 0`, `NoCrack = 1`), so the sigmoid output is the probability of `NoCrack`. The prediction function handles this mapping.

## Model Architecture

| Stage | Layers |
|-------|--------|
| Conv Block 1 | Conv2D(32) → BatchNorm → MaxPool |
| Conv Block 2 | Conv2D(64) → BatchNorm → MaxPool |
| Conv Block 3 | Conv2D(128) → BatchNorm → MaxPool |
| Conv Block 4 | Conv2D(256) → BatchNorm → MaxPool |
| Classifier | Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid) |

Input: 128 × 128 × 3 | Optimizer: Adam | Loss: Binary cross-entropy

## Installation

```bash
git clone https://github.com/<your-username>/crack-detection-cnn.git
cd crack-detection-cnn

python -m venv MarvellousCNN
MarvellousCNN\Scripts\activate        # Windows
# source MarvellousCNN/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

## Usage

```bash
python Marvellous_CNN_Surface_Crack_Detection.py
```

The script will:
1. Validate the dataset and create the train/validation/test split
2. Train the CNN with augmentation and callbacks
3. Plot accuracy and loss curves
4. Evaluate on the test set and print the confusion matrix and classification report
5. Save `Best_Crack_Detection_Model.keras` and `Final_Marvellous_Crack_Detection_Model.keras`
6. Run a sample single-image prediction

To predict on your own image:

```python
predict_single_image("path/to/your_image.jpg")
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Batch size | 32 |
| Max epochs | 15 |
| Early stopping | patience 4 on `val_loss` |
| LR reduction | factor 0.2, patience 2, min LR 1e-5 |
| Random seed | 42 |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | XX.XX% |
| Test Loss | X.XXXX |

## Project Structure

```
.
├── Marvellous_CNN_Surface_Crack_Detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements
- Transfer learning (MobileNetV2, ResNet50)
- Crack segmentation with U-Net
- Grad-CAM visualizations
- Web app deployment (Streamlit/Flask)

## Author
**Apurv Ranjane**
[GitHub](https://github.com/ApurvRanjane)
