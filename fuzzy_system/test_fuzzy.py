import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from fuzzy_system.fuzzy_layer import fuzzy_transform


features = torch.randn(10, 1024)

params = [0.3, 0.2, 0.7, 0.2]

fuzzy_features = fuzzy_transform(features, params)

print("Original shape:", features.shape)
print("Fuzzy shape:", fuzzy_features.shape)