"""网络定义：用于 Fashion-MNIST 服饰图像分类。

包含两种模型，用于“一次代码修改”的对照实验：
    修改前（baseline） ：多层感知机 MLP —— 把图像展平为一维向量后接全连接层
    修改后（optimized）：卷积神经网络 CNN —— 用卷积层提取局部特征，并加入批归一化

改进点：CNN 通过卷积核共享参数、提取图像局部空间特征，比 MLP 更擅长图像分类。
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机（baseline）：28x28 展平为 784 -> 256 -> 128 -> 10。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FashionCNN(nn.Module):
    """卷积神经网络（改进版）：卷积 + 批归一化 + 池化 + 全连接。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    dummy = torch.randn(8, 1, 28, 28)
    for name, cls in [("MLP", MLP), ("CNN", FashionCNN)]:
        out = cls()(dummy)
        print(f"{name} 输出形状:", tuple(out.shape))
        assert out.shape == (8, 10), "输出形状不正确"
    print("model.py 自检通过")
