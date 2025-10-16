#Abstract Convolutional Nueral Network Class

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CNNConfig:
    input_channels: int
    num_classes: int
    conv_layers: Tuple[int, int, int]  # (out_channels, kernel_size, stride)
    dropout: Optional[float] = None

class AbstractCNN(nn.Module, ABC):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        self.layers: nn.Sequential
        self.classifier: nn.Module
        self.dropout: nn.Module = nn.Identity()
        self.build()

    @abstractmethod
    def build(self) -> nn.Sequential:
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        if self.dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x
    

class SimpleCNN(AbstractCNN):
    def build(self) -> nn.Sequential:
        c = self.config
        w1, w2, w3 = c.conv_layers

        #Block 1 28x28 -> 14x14
        self.b1 = nn.Sequential(
            nn.Conv2d(c.input_channels, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w1, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        #Block 2 14x14 -> 7x7
        self.b2 = nn.Sequential(
            nn.Conv2d(w1, w2, 3, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w2, w2, 3, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        #Feature Head 7x7 -> 7x7
        self.head = nn.Sequential(
            nn.Conv2d(w2, w3, 3, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True)
        )

        # Combine all layers
        self.layers = nn.Sequential(
            self.b1,
            self.b2,
            self.head
        )

        # Classifier
        if c.dropout:
            self.dropout = nn.Dropout(c.dropout)
        self.classifier = nn.Linear(w3, c.num_classes)

if __name__ == "__main__":
    cfg = CNNConfig(input_channels=1, num_classes=26, conv_layers=(32, 64, 128), dropout=0.3)
    model = SimpleCNN(cfg)
    x = torch.randn(8, 1, 28, 28)        # batch of 8, preprocessed images
    y = model(x)
    print("logits shape:", y.shape)      # expect [8, 26]
    print("param count:", sum(p.numel() for p in model.parameters()))
       
