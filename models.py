# models.py

import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """
    CIFAR-10 简单卷积神经网络
    后面作为 Model B：目标模型
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18_cifar10(num_classes=10):
    """
    CIFAR-10 版本 ResNet18
    后面作为 Model A：源模型
    """
    try:
        model = resnet18(weights=None)
    except TypeError:
        model = resnet18(pretrained=False)

    # CIFAR-10 是 32x32 小图，所以把原 ResNet 的 7x7 卷积改成 3x3
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # 去掉 ImageNet 版本的 maxpool
    model.maxpool = nn.Identity()

    # 修改分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def build_model(model_name):
    """
    根据名称构建模型
    """
    if model_name == "resnet18":
        return build_resnet18_cifar10(num_classes=10)
    elif model_name == "simplecnn":
        return SimpleCNN(num_classes=10)
    else:
        raise ValueError(f"未知模型名称: {model_name}")


def load_model(model_name, checkpoint_path, device):
    """
    后面攻击实验会用到：加载训练好的模型
    """
    model = build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model