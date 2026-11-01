"""数据加载与预处理（Fashion-MNIST）。

Fashion-MNIST 数据集首次运行会自动下载到 ./data 目录（需要联网）。
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Fashion-MNIST 各类别英文名（用于画图，避免中文字体缺失导致的乱码）
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Fashion-MNIST 训练集像素均值 / 标准差（经验值），用于归一化
MEAN = 0.2860
STD = 0.3530


def get_transform():
    """返回统一的图像预处理：ToTensor + Normalize（无数据增强）。"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])


def get_loaders(data_dir: str = "./data", batch_size: int = 64,
                num_workers: int = 0, val_ratio: float = 0.1, seed: int = 42):
    """构建训练 / 验证 / 测试 DataLoader。

    训练集按 val_ratio 划分出验证集；训练、验证、测试使用相同的预处理。
    """
    transform = get_transform()

    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader
