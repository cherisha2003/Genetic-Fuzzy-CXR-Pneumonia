import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def train_classifier(features, labels):

    model = SVC(
        kernel="rbf",
        probability=True
    )

    model.fit(features, labels)

    return model


def evaluate(model, features, labels):

    preds = model.predict(features)

    probs = model.predict_proba(features)[:, 1]

    acc = accuracy_score(labels, preds)

    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(labels, preds))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix.png")

    plt.close()


    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("results/roc_curve.png")

    plt.close()