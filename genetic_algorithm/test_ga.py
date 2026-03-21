import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from genetic_algorithm.ga_optimizer import run_ga


features = torch.randn(100, 1024)

labels = np.random.randint(0, 2, 100)

best_params = run_ga(features, labels)

print("Best parameters found:")
print(best_params)