import torch
import torch.nn as nn
from torchvision import models

# ------------------------------
# Build the model
# ------------------------------
def build_model(num_classes: int):
    """
    Creates a ResNet18 model for classification.
    """
    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
