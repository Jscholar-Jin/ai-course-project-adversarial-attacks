import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward_head(self, features):
        return self.head(features)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        try:
            backbone = resnet18(weights=None)
        except TypeError:
            backbone = resnet18(pretrained=False)

        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward_features(self, x):
        return self.backbone(x)

    def forward_head(self, features):
        return self.classifier(features)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def get_model(name):
    key = name.lower()
    if key == "resnet":
        return ResNetCIFAR()
    if key == "cnn":
        return CNN()
    raise ValueError(f"unknown model: {name}")


def load_model(name, path, device):
    model = get_model(name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
