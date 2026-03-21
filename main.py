import sys
import os

sys.path.append(os.path.abspath("."))

import numpy as np
import torch

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from genetic_algorithm.ga_optimizer import run_ga
from fuzzy_system.fuzzy_layer import fuzzy_transform
from training.train_classifier import train_classifier, evaluate


print("Loading extracted features...")

features = torch.tensor(np.load("data/train_features.npy"))
labels = np.load("data/train_labels.npy")


print("Running Genetic Algorithm optimization...")

best_params = run_ga(features, labels)

print("Best fuzzy parameters:", best_params)


print("Applying fuzzy transformation...")

fuzzy_features = fuzzy_transform(features, best_params)


print("Starting 5-Fold Cross Validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
accuracies = []


for train_index, test_index in kf.split(fuzzy_features):

    print(f"\nFold {fold}")

    X_train = fuzzy_features[train_index]
    X_test = fuzzy_features[test_index]

    y_train = labels[train_index]
    y_test = labels[test_index]

    model = train_classifier(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Fold Accuracy:", acc)

    accuracies.append(acc)

    fold += 1


print("\nCross Validation Results")

print("Accuracies:", accuracies)

print("Mean Accuracy:", np.mean(accuracies))


print("\nRunning final evaluation on last fold...")

evaluate(model, X_test, y_test)


print("Pipeline completed successfully.")