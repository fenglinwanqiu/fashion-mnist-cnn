# 基于卷积神经网络（CNN）的 Fashion-MNIST 服饰图像分类

> 图像处理课程大作业
>
> 核心内容：用 **PyTorch** 完成服饰图像分类，并通过 **一次代码修改（用卷积神经网络 CNN 替代多层感知机 MLP）** 对比改进前后的结果。

---

## 0. GitHub 链接

```
https://github.com/fenglinwanqiu/fashion-mnist-cnn
```

---

## 1. 任务描述

本作业实现一个基于**卷积神经网络（CNN）**的图像分类系统，对 Fashion-MNIST 数据集中的 10 类服饰图像进行自动分类。具体任务包括：

1. 使用 PyTorch 搭建并训练图像分类模型；
2. 设计一组**对照实验**：对比「多层感知机 MLP」与「卷积神经网络 CNN」两种模型结构；
3. 从**测试集准确率、损失曲线、混淆矩阵**三个维度对比结果，说明 CNN 对图像分类任务的优势。

---

## 2. 数据描述

采用公开数据集 **Fashion-MNIST**（Zalando 发布的服饰图像数据集）：

| 项目 | 说明 |
| --- | --- |
| 来源 | Fashion-MNIST（MNIST 的服饰替代版） |
| 规模 | 训练集 60,000 张，测试集 10,000 张 |
| 图像 | 28×28 灰度图（单通道） |
| 类别 | 10 类：T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot |

**预处理**：统一使用 `ToTensor()` 转张量 + `Normalize(均值=0.286, 标准差=0.353)` 归一化，两个对比模型使用完全相同的数据。

数据由 `torchvision` 自动下载到 `./data` 目录（首次运行需联网）。

---

## 3. 代码修改描述

代码修改位于 `model.py`，即**用卷积神经网络（CNN）替代多层感知机（MLP）**。

**修改前（baseline，多层感知机 MLP）** —— 把整张图像展平为一维向量后接全连接层，丢失了空间结构：

```python
class MLP(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                      # 28x28 -> 784 的一维向量
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )
```

**修改后（optimized，卷积神经网络 CNN）** —— 用卷积层提取图像局部特征，并加入批归一化：

```python
class FashionCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
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
```

**修改目的**：MLP 把图像当成无结构的一维向量，忽略了像素之间的空间关系；CNN 通过卷积核**共享参数、提取局部空间特征**，更擅长图像分类，配合批归一化可加速收敛、提升精度。两个模型使用完全相同的超参数、随机种子和数据集划分，保证对比公平，唯一变量就是网络结构（MLP vs CNN）。

---

## 4. 结果对比

> 以下为 `python train.py` 在固定随机种子（seed=42）下自动生成的对比结果。

### 4.1 测试集指标对比

| 版本 | 测试集准确率 | 测试集损失 |
| --- | --- | --- |
| 基线（多层感知机 MLP） | 89.80% | 0.3092 |
| 优化（卷积神经网络 CNN） | 92.68% | 0.2809 |

测试准确率提升：**+2.88%**

### 4.2 准确率对比图

![测试准确率对比](results/comparison.png)

### 4.3 训练曲线（左：损失，右：准确率）

多层感知机 MLP：

![MLP 训练曲线](results/mlp_curves.png)

卷积神经网络 CNN：

![CNN 训练曲线](results/cnn_curves.png)

### 4.4 混淆矩阵

多层感知机 MLP：

![MLP 混淆矩阵](results/mlp_cm.png)

卷积神经网络 CNN：

![CNN 混淆矩阵](results/cnn_cm.png)

### 4.5 结论

> 卷积神经网络 CNN 的测试准确率为 92.68%，相比多层感知机 MLP 的 89.80% 提升了 **+2.88%**，且测试损失更低（0.2809 vs 0.3092）。这说明 CNN 通过卷积核提取图像的局部空间特征，比把图像当作一维向量的 MLP 更适合图像分类任务。

---

## 5. 环境与运行方法

### 5.1 环境

- Python 3.8+
- 依赖见 `requirements.txt`

### 5.2 安装依赖

```bash
pip install -r requirements.txt
```

> 若只装 CPU 版 PyTorch（无 GPU），可参考 PyTorch 官网的 CPU 安装命令，例如：
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### 5.3 运行

```bash
python train.py
```

运行完成后，所有结果保存在 `results/` 目录。

> 训练参数（轮数、批大小、学习率等）可在 `train.py` 的 `main()` 中的 `config` 字典里修改。

---

## 6. 项目结构

```
.
├── train.py             # 主脚本：训练 MLP + CNN，并生成对比结果
├── model.py             # 模型定义（含“代码修改点”：MLP 与 CNN）
├── data_utils.py        # 数据加载与预处理
├── requirements.txt     # 依赖列表
├── .gitignore           # 忽略 data/、缓存等
├── README.md            # 本文件
├── report.docx          # Word 报告
├── report.pdf           # PDF 报告
├── data/                # 自动下载的数据集
└── results/             # 运行后生成的结果图与指标
```

---

## 7. 复现与公平性说明

- 两个模型共用相同的随机种子（`seed=42`）、优化器（Adam，`lr=1e-3`）、批大小（64）和轮数（20）；
- 训练/验证/测试集划分完全相同，数据预处理一致；
- 因此两个模型之间**唯一的区别就是网络结构（MLP vs CNN）**，结果差异可完全归因于这一处代码修改。
