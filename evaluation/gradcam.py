import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_gradcam(image, model):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    target_layer.register_forward_hook(forward_hook)

    target_layer.register_backward_hook(backward_hook)

    output = model(image)

    class_idx = output.argmax()

    model.zero_grad()

    output[0, class_idx].backward()

    grad = gradients[0].cpu().data.numpy()[0]

    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224,224))

    cam = cam - cam.min()

    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    return heatmap