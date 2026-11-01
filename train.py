"""
Fashion-MNIST 服饰图像分类：多层感知机（MLP）vs 卷积神经网络（CNN）对比实验。

用法：
    python train.py

运行后会在 results/ 目录下自动生成：
    mlp_history.json        MLP 每轮训练/验证指标
    cnn_history.json        CNN 每轮训练/验证指标
    mlp_curves.png          MLP 损失与准确率曲线
    cnn_curves.png          CNN 损失与准确率曲线
    mlp_cm.png              MLP 测试集混淆矩阵
    cnn_cm.png              CNN 测试集混淆矩阵
    comparison.png          两个模型测试准确率对比柱状图
    summary.md              结果对比摘要（可直接复制进 README）

核心“代码修改”点见 model.py：
    修改前（baseline） ：多层感知机 MLP（全连接）
    修改后（optimized）：卷积神经网络 CNN（卷积 + 批归一化）
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # 无图形界面环境也能保存图片
import matplotlib.pyplot as plt

from model import MLP, FashionCNN
from data_utils import CLASSES, get_loaders

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"


def set_seed(seed: int = 42):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回 (平均损失, 准确率)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估，返回 (平均损失, 准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    """返回整个 loader 上的预测标签与真实标签。"""
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds.append(outputs.argmax(1).cpu())
        targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def plot_curves(history, name, path):
    """画损失与准确率曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{name} - Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(targets, preds, name, path):
    """画测试集混淆矩阵。"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - Test Confusion Matrix")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(results, path):
    """画两个模型测试准确率对比柱状图。"""
    labels = ["MLP (baseline)", "CNN"]
    accs = [results["mlp"]["test_acc"], results["cnn"]["test_acc"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=["#5b9bd5", "#ed7d31"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc * 100:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(results, path):
    mlp = results["mlp"]
    cnn = results["cnn"]
    delta = cnn["test_acc"] - mlp["test_acc"]
    lines = [
        "# 结果对比摘要",
        "",
        "| 版本 | 测试集准确率 | 测试集损失 |",
        "| --- | --- | --- |",
        f"| 基线（多层感知机 MLP） | {mlp['test_acc'] * 100:.2f}% | {mlp['test_loss']:.4f} |",
        f"| 优化（卷积神经网络 CNN） | {cnn['test_acc'] * 100:.2f}% | {cnn['test_loss']:.4f} |",
        "",
        f"测试准确率提升：**{delta * 100:+.2f}%**",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment(name, model_cls, device, config):
    """训练并评估一个模型，返回结果字典。"""
    set_seed(config["seed"])  # 每个模型前重置种子，保证初始权重一致、对比公平
    train_loader, val_loader, test_loader = get_loaders(
        data_dir=str(DATA_DIR),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"[{name}] Epoch {epoch:02d}/{config['epochs']:02d} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        scheduler.step()

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[{name}] Test: loss={test_loss:.4f} accuracy={test_acc * 100:.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump({**history, "test_loss": test_loss, "test_acc": test_acc},
                  f, indent=2, ensure_ascii=False)

    plot_curves(history, name, str(RESULTS_DIR / f"{name}_curves.png"))
    preds, targets = predict_all(model, test_loader, device)
    plot_confusion(targets, preds, name, str(RESULTS_DIR / f"{name}_cm.png"))

    return {"name": name, "test_loss": test_loss, "test_acc": test_acc,
            "history": history}


def main():
    config = {
        "epochs": 20,        # 训练轮数
        "batch_size": 64,    # 批大小
        "lr": 1e-3,          # 学习率
        "num_workers": 0,    # Windows 下建议保持 0
        "seed": 42,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    results = {}
    for name, model_cls in [("mlp", MLP), ("cnn", FashionCNN)]:
        print(f"\n===== 开始训练: {name} =====")
        results[name] = run_experiment(name, model_cls, device, config)

    plot_comparison(results, str(RESULTS_DIR / "comparison.png"))
    write_summary(results, str(RESULTS_DIR / "summary.md"))
    print("\n全部完成！结果已保存到 results/ 目录。")


if __name__ == "__main__":
    main()
