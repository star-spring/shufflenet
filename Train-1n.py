# import os
# import math
# import argparse

# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from sklearn.model_selection import train_test_split

# from shufflenet1d import ShuffleNetV2_1D  # 1D ShuffleNet
# from utils import train_one_epoch, evaluate  # 保留原来的训练/验证函数


# # ---------------- 自定义 1D 数据集 ----------------
# class My1DDataset(Dataset):
#     def __init__(self, data_paths, labels):
#         self.data_paths = data_paths
#         self.labels = labels

#     def __getitem__(self, idx):
#         x = np.load(self.data_paths[idx])  # shape: (N,)
#         x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # -> [1, N]
#         y = torch.tensor(self.labels[idx], dtype=torch.long)
#         return x, y

#     def __len__(self):
#         return len(self.data_paths)


# # ---------------- 主函数 ----------------
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     tb_writer = SummaryWriter()
#     os.makedirs("./weightss_1d", exist_ok=True)

#     # ---------------- 读取 normal/abnormal 数据 ----------------
#     normal_folder = os.path.join(args.data_path, "coherence_samples")
#     abnormal_folder = os.path.join(args.data_path, "ab-coherence_samples")

#     X, y = [], []

#     for f in os.listdir(normal_folder):
#         if f.endswith(".npy"):
#             X.append(os.path.join(normal_folder, f))
#             y.append(0)

#     for f in os.listdir(abnormal_folder):
#         if f.endswith(".npy"):
#             X.append(os.path.join(abnormal_folder, f))
#             y.append(1)

#     # ---------------- 划分训练/验证集 ----------------
#     train_data, val_data, train_labels, val_labels = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
     
#     print(f"训练集样本数: {len(train_data)}")
#     print(f"验证集样本数: {len(val_data)}")

#     train_dataset = My1DDataset(train_data, train_labels)
#     val_dataset = My1DDataset(val_data, val_labels)

#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#     print(f'Using {nw} dataloader workers per process')

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

#     # ---------------- 初始化 1D ShuffleNet ----------------
#     model = ShuffleNetV2_1D(num_classes=args.num_classes, input_channels=1).to(device)

#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             if "fc" not in name:
#                 para.requires_grad_(False)

#     # ---------------- 优化器和学习率调度 ----------------
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4e-5)
#     lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

#     # ---------------- 训练循环 ----------------
#     for epoch in range(args.epochs):
#         mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
#         scheduler.step()

#         acc = evaluate(model, val_loader, device)

#         print(f"[epoch {epoch}] accuracy: {round(acc, 3)}")
#         tb_writer.add_scalar("loss", mean_loss, epoch)
#         tb_writer.add_scalar("accuracy", acc, epoch)
#         tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

#         torch.save(model.state_dict(), f"./weightss_1d/model-{epoch}.pth")


# # ---------------- argparse ----------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=2)
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=0.005)
#     parser.add_argument('--lrf', type=float, default=0.05)
#     parser.add_argument('--data-path', type=str,
#                         default=r"E:\BaiduNetdisk\Fault-diagnosis-based-on-deep-learning-main ResDSC\cwt_picture\CF")
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:0')

#     opt = parser.parse_args()
#     main(opt)


import os
import math
import argparse
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from shufflenet1d import ShuffleNetV2_1D  # 1D ShuffleNet
from utils import train_one_epoch, evaluate  # 保留原来的训练/验证函数


# ---------------- 自定义 1D 数据集 ----------------
class My1DDataset(Dataset):
    def __init__(self, data_paths, labels):
        self.data_paths = data_paths
        self.labels = labels

    def __getitem__(self, idx):
        x = np.load(self.data_paths[idx])  # shape: (N,)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # -> [1, N]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data_paths)


# ---------------- 主函数 ----------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tb_writer = SummaryWriter()
    os.makedirs("./weightss_1d", exist_ok=True)

    # ---------------- 读取 normal/abnormal 数据 ----------------
    normal_folder = os.path.join(args.data_path, "coherence_samples")
    abnormal_folder = os.path.join(args.data_path, "ab-coherence_samples")

    X, y = [], []

    for f in os.listdir(normal_folder):
        if f.endswith(".npy"):
            X.append(os.path.join(normal_folder, f))
            y.append(0)

    for f in os.listdir(abnormal_folder):
        if f.endswith(".npy"):
            X.append(os.path.join(abnormal_folder, f))
            y.append(1)

    # ---------------- 划分训练/验证/测试集 (7:2:1) ----------------
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=1/3, random_state=42, stratify=temp_labels
    )

    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")

    # ---------------- 保存测试集文件 ----------------
    # ---------------- 保存测试集文件 ----------------
    test_save_root = os.path.join(args.data_path, "test_split")
    normal_save_path = os.path.join(test_save_root, "normal")
    abnormal_save_path = os.path.join(test_save_root, "abnormal")

    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(abnormal_save_path, exist_ok=True)

    for path, label in zip(test_data, test_labels):
        fname = os.path.basename(path)
        dst_dir = normal_save_path if label == 0 else abnormal_save_path
        shutil.copy(path, os.path.join(dst_dir, fname))

    print(f"✅ 测试集样本已保存到: {test_save_root}")


    # ---------------- 创建 DataLoader ----------------
    train_dataset = My1DDataset(train_data, train_labels)
    val_dataset = My1DDataset(val_data, val_labels)
    test_dataset = My1DDataset(test_data, test_labels)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    # ---------------- 初始化 1D ShuffleNet ----------------
    model = ShuffleNetV2_1D(num_classes=args.num_classes, input_channels=1).to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    # ---------------- 优化器和学习率调度 ----------------
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ---------------- 训练循环 ----------------
    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        acc = evaluate(model, val_loader, device)

        print(f"[epoch {epoch}] accuracy: {round(acc, 3)}")
        tb_writer.add_scalar("loss", mean_loss, epoch)
        tb_writer.add_scalar("accuracy", acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), f"./weightss_1d/model-{epoch}.pth")

    # ---------------- 测试集评估 ----------------
    print("\n开始评估测试集...")
    test_acc = evaluate(model, test_loader, device)
    print(f"🎯 测试集准确率: {round(test_acc, 3)}")


# ---------------- argparse ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lrf', type=float, default=0.05)
    parser.add_argument('--data-path', type=str,
                        default=r"E:\BaiduNetdisk\Fault-diagnosis-based-on-deep-learning-main ResDSC\cwt_picture\CF")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')

    opt = parser.parse_args()
    main(opt)
