from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class WFRNet(nn.Module):
    def __init__(self, number_of_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_size = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_size, number_of_classes)

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        features = self.backbone(inputs)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

