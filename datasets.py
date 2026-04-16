"""
Oxford Flowers102 数据加载。
torchvision 自带，第一次运行会自动下载 (~330MB)。
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_dataloaders(root="./data", batch_size=64, num_workers=4):
    train_tf, val_tf = build_transforms()

    # Flowers102 官方划分: train(1020) / val(1020) / test(6149)
    # 这里把 train+val 合起来当训练集, test 当评估集 —— 样本更多, 更稳
    train_set_a = datasets.Flowers102(root=root, split="train", download=True, transform=train_tf)
    train_set_b = datasets.Flowers102(root=root, split="val",   download=True, transform=train_tf)
    from torch.utils.data import ConcatDataset
    train_set = ConcatDataset([train_set_a, train_set_b])

    test_set = datasets.Flowers102(root=root, split="test", download=True, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"train samples: {len(train_set)}, val samples: {len(test_set)}")
    return train_loader, val_loader


if __name__ == "__main__":
    tr, va = build_dataloaders(batch_size=4, num_workers=0)
    x, y = next(iter(tr))
    print("batch:", x.shape, y.shape, "label range:", y.min().item(), y.max().item())
