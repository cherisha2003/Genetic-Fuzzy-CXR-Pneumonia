# Genetic-Optimized Fuzzy Feature Fusion for Chest X-Ray Pneumonia Classification using DenseNet-121 Embeddings

## Overview

This project presents a hybrid artificial intelligence framework for pneumonia detection from chest X-ray images. The system integrates deep learning feature extraction, fuzzy logic-based feature fusion, and genetic algorithm optimization to improve classification performance and robustness in medical image analysis.

The proposed pipeline combines DenseNet-121 embeddings with a fuzzy membership-based feature representation. A genetic algorithm is used to automatically optimize the fuzzy membership parameters, allowing the model to better capture uncertainty present in medical imaging data.

## System Architecture

The complete pipeline of the proposed system is illustrated below:

Chest X-ray Image  
↓  
Image Preprocessing  
↓  
DenseNet-121 Feature Extraction  
↓  
Feature Embeddings (1024-dim)  
↓  
Fuzzy Feature Fusion Layer  
↓  
Genetic Algorithm Optimization  
↓  
Classifier (Logistic Regression)  
↓  
Pneumonia Classification (Normal vs Pneumonia)

## Key Components

### 1. DenseNet-121 Feature Extraction
A pretrained DenseNet-121 model is used to extract deep feature embeddings from chest X-ray images. The final classification layer is removed, producing a 1024-dimensional feature vector representing the visual characteristics of lung structures.

### 2. Fuzzy Feature Fusion
To model uncertainty and variability in medical images, fuzzy membership functions are applied to the extracted features. Gaussian membership functions convert raw feature values into fuzzy representations such as low and high memberships.

### 3. Genetic Algorithm Optimization
A genetic algorithm is used to optimize the fuzzy membership parameters (means and standard deviations). The algorithm evolves candidate solutions over multiple generations to find parameter values that maximize classification performance.

### 4. Classification
The optimized fuzzy feature representations are used to train a logistic regression classifier to distinguish between normal and pneumonia X-ray images.

## Dataset

This project uses the **Chest X-Ray Pneumonia Dataset** containing labeled chest radiographs categorized as:

- Normal
- Pneumonia

Dataset structure:

data/chest_xray  
├── train  
├── val  
└── test  

Images are resized to **224 × 224** and normalized using ImageNet statistics before feature extraction.

## Evaluation Metrics

The system performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve (AUC)

Visualization results include:

- Confusion Matrix
- ROC Curve
- Grad-CAM heatmaps highlighting pneumonia regions in X-ray images

## Project Structure
Genetic-Fuzzy-CXR-Pneumonia
│
├── preprocessing
│ └── preprocess.py
│
├── feature_extraction
│ ├── densenet_features.py
│ └── extract_features.py
│
├── fuzzy_system
│ └── fuzzy_layer.py
│
├── genetic_algorithm
│ └── ga_optimizer.py
│
├── training
│ └── train_classifier.py
│
├── evaluation
│ └── gradcam.py
│
├── results
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ └── gradcam_result.png
│
├── main.py
├── requirements.txt
└── README.md


## Installation

Clone the repository:


git clone https://github.com/yourusername/Genetic-Fuzzy-CXR-Pneumonia.git

cd Genetic-Fuzzy-CXR-Pneumonia


Create virtual environment:


python3 -m venv venv
source venv/bin/activate


Install dependencies:


pip install -r requirements.txt


## Running the Project

Extract DenseNet embeddings:


python feature_extraction/extract_embeddings.py


Run the full pipeline:


python main.py


## Explainability with Grad-CAM

Grad-CAM visualization is implemented to highlight regions in chest X-ray images that contribute most to the model’s prediction. This provides interpretability for medical AI systems and helps validate that the model focuses on relevant lung areas.

Example output is stored in:


results/gradcam_result.png


## Future Improvements

Possible extensions include:

- Replacing logistic regression with a deep neural classifier
- Using additional fuzzy membership functions
- Hyperparameter optimization for genetic algorithm
- Cross-validation experiments
- Testing on additional medical imaging datasets

## Author

Cherisha KG  
Master’s Student — Computer Science
