import torch
from tqdm import tqdm


def extract_features(model, dataloader):

    features = []
    labels = []

    with torch.no_grad():

        for images, targets in tqdm(dataloader):

            outputs = model(images)

            features.append(outputs)
            labels.append(targets)

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features, labels