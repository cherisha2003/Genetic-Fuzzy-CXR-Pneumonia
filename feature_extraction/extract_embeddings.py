import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from preprocessing.preprocess import get_dataloaders
from feature_extraction.densenet_features import get_feature_extractor
from feature_extraction.extract_features import extract_features


print("Loading dataset...")

train_loader, val_loader, test_loader = get_dataloaders("data/chest_xray")

print("Loading DenseNet121...")

model = get_feature_extractor()

print("Extracting training features...")

train_features, train_labels = extract_features(model, train_loader)

print("Saving features...")

np.save("data/train_features.npy", train_features.numpy())
np.save("data/train_labels.npy", train_labels.numpy())

print("Done!")