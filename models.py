"""
统一的模型构造接口。
直接用 torchvision 官方模型，因为只有官方结构能加载 ImageNet 预训练权重。
"""
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()

    if arch == "resnext50":
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"unknown arch: {arch}")

    return model


if __name__ == "__main__":
    import torch
    for arch in ["resnext50", "densenet121"]:
        for pre in [False, True]:
            m = build_model(arch, num_classes=102, pretrained=pre)
            x = torch.randn(2, 3, 224, 224)
            print(arch, "pretrained=", pre, "->", m(x).shape)
