import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import torchvision.transforms as transforms
import torchvision.models as models

from evaluation.gradcam import generate_gradcam


model = models.densenet121(pretrained=True)
model.eval()


image = cv2.imread("data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")

image = cv2.resize(image, (224,224))

transform = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = transform(image).unsqueeze(0)


heatmap = generate_gradcam(image_tensor, model)

overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

cv2.imwrite("results/gradcam_result.png", overlay)

print("Grad-CAM saved to results/")