# 从头训练 vs 微调  对比实验 (ResNeXt & DenseNet)

CNN 课程作业：在 Oxford Flowers102 数据集上对比 **从头训练 (scratch)** 与 **ImageNet 预训练微调 (finetune)** 两种策略，使用 ResNeXt-50 和 DenseNet-121 两个模型。

## 实验设置

| 编号 | 模型 | 初始化 | 学习率 |
|---|---|---|---|
| A1 | ResNeXt-50 (32×4d) | 随机 | 1e-2 |
| A2 | ResNeXt-50 (32×4d) | ImageNet 预训练 | 1e-3 (head ×10) |
| B1 | DenseNet-121 | 随机 | 1e-2 |
| B2 | DenseNet-121 | ImageNet 预训练 | 1e-3 (head ×10) |

- 数据集: **Oxford Flowers102** (102 类, train+val=2040 张, test=6149 张)
- 输入: 224×224, ImageNet mean/std 归一化
- 数据增强: RandomResizedCrop + HorizontalFlip + ColorJitter
- 优化器: SGD (momentum=0.9, weight_decay=1e-4)
- 调度器: CosineAnnealingLR
- 损失: CrossEntropyLoss
- Epochs: 30 (默认), Batch size: 64
- 控制变量: 数据/增强/优化器/epoch 全部相同, 仅初始化方式与学习率不同

## 文件结构

```
finetune-vs-scratch/
├── README.md
├── requirements.txt
├── datasets.py        # Flowers102 数据加载
├── models.py          # ResNeXt50 / DenseNet121 构造器
├── train.py           # 训练主脚本 (参数化)
├── plot.py            # 读 logs 画对比图
├── run_all.sh         # 一键跑 4 组实验
├── logs/              # 每组实验的训练日志 csv
├── checkpoints/       # 每组实验的最优权重
└── figures/           # 对比图 (val_acc / val_loss / overfit)
```

## 在 AutoDL 上运行

1. 租实例时镜像选 **PyTorch 2.x / Python 3.10 / CUDA 11.8** 之类的官方镜像
2. 打开 JupyterLab 终端:
   ```bash
   git clone <你的 git 地址>
   cd finetune-vs-scratch
   pip install -r requirements.txt   # 大概率只缺 matplotlib
   ```
3. 跑通单组 (调试):
   ```bash
   python train.py --exp A2 --epochs 2
   ```
4. 跑全部 4 组:
   ```bash
   bash run_all.sh
   ```
   或自定义:
   ```bash
   EPOCHS=40 BS=64 bash run_all.sh
   ```
5. 画图:
   ```bash
   python plot.py
   ```

数据集会在第一次运行时自动下载到 `./data/flowers-102/` (约 330MB)。

## 本地 (CPU) 调试

```bash
pip install torch torchvision matplotlib
python train.py --exp A2 --epochs 1 --batch_size 4 --num_workers 0
```
能跑通就 push 到 git, 然后到云端正式训练。

## 输出示例

训练完成后:
- `logs/A1.csv` ~ `logs/B2.csv`: 每个 epoch 的 train/val loss & acc
- `figures/val_acc.png`: 4 条曲线对比验证集准确率
- `figures/val_loss.png`: 验证集 loss 对比
- `figures/overfit.png`: 每组实验 train vs val 准确率, 看过拟合程度
- 终端打印汇总表 (best/final acc, 总用时)

## 报告里要写的内容 (PDF 提交)

1. **实验目的**: 比较 scratch / finetune 在小数据集细粒度分类上的差异
2. **实验设置**: 上面那张表 + 数据集介绍
3. **结果**:
   - 4 组的最终/最佳准确率 (表格)
   - 验证集准确率收敛曲线 (figures/val_acc.png)
   - 训练 vs 验证准确率 (figures/overfit.png) → 过拟合分析
   - 单 epoch 用时 / 达到某阈值所需 epoch 数
4. **分析**:
   - 为什么微调收敛更快、准确率更高?
   - 从头训练在小数据集上为什么容易过拟合?
   - 两个模型 (ResNeXt vs DenseNet) 的差异?
5. **结论**
6. **代码地址**: 你的 git 仓库 URL

## Git 上传

```bash
cd finetune-vs-scratch
git init
git add .
git commit -m "finetune vs scratch experiment"
git remote add origin <你的仓库地址>
git push -u origin main
```

注意 `.gitignore` 一下 `data/`, `logs/`, `checkpoints/` 之类的大文件夹。
