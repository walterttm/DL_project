# Finetune vs Scratch on Flowers102 (ResNeXt & DenseNet)

##  项目简介

本项目用于比较 **从头训练（scratch）** 与 **预训练微调（finetune）** 在小规模数据集上的性能差异。

实验基于：

* ResNeXt-50
* DenseNet-121
* 数据集：Oxford Flowers102（102类）

---

##  实验目的

研究在小数据集场景下：

* 从头训练 vs 迁移学习（微调）的性能差异
* 不同CNN结构（ResNeXt vs DenseNet）的表现对比

---

##  实验设置

| 编号 | 模型          | 初始化方式 | 学习率  |
| -- | ----------- | ----- | ---- |
| A1 | ResNeXt50   | 从头训练  | 1e-2 |
| A2 | ResNeXt50   | 预训练微调 | 1e-3 |
| B1 | DenseNet121 | 从头训练  | 1e-2 |
| B2 | DenseNet121 | 预训练微调 | 1e-3 |

其他设置：

* 输入尺寸：224×224
* 数据增强：RandomResizedCrop + Flip + ColorJitter
* 优化器：SGD + momentum
* 损失函数：CrossEntropyLoss
* 学习率调度：CosineAnnealing

---

##  实验结果

| 模型          | 训练方式 | 验证集准确率    |
| ----------- | ---- | --------- |
| ResNeXt50   | 从头训练 | 40.2%     |
| ResNeXt50   | 微调   | **91.3%** |
| DenseNet121 | 从头训练 | 42.3%     |
| DenseNet121 | 微调   | **94.9%** |

---

##  可视化结果

训练过程中生成如下曲线：

* 验证集准确率曲线（val_acc）
* 验证集损失曲线（val_loss）
* 训练 vs 验证准确率（过拟合分析）

结果图保存在：

```bash
figures/
```

---

##  结果分析

1. **微调显著优于从头训练**

* 从头训练仅达到约 40% 准确率
* 微调可达到 90%+
* 提升超过 50%

 说明预训练特征对小数据集非常重要

---

2. **收敛速度差异明显**

* 微调模型在第1轮即可达到较高精度
* 从头训练需要多个epoch才能收敛

---

3. **模型结构对比**

* DenseNet略优于ResNeXt
* 可能由于特征复用机制更高效

---

4. **过拟合情况**

* 从头训练：欠拟合（能力不足）
* 微调模型：泛化能力更强

---

##  运行方式

### 1️ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️ 单个实验

```bash
python train.py --exp A2
```

---

### 3️ 运行全部实验

```bash
bash run_all.sh
```

---

### 4️ 绘图

```bash
python plot.py
```

---

##  项目结构

```
finetune_vs_scratch/
├── train.py
├── models.py
├── datasets.py
├── plot.py
├── run_all.sh
├── requirements.txt
├── figures/
└── logs/
```

---

##  数据集

* Oxford Flowers102
* 自动下载或手动放置至：

```
data/flowers-102/
```

---

##  结论

在小规模数据集上：

 **迁移学习（微调）远优于从头训练**

 DenseNet 表现略优于 ResNeXt

---

##  代码地址

 https://gitee.com/tantauming/dl_pro

---


