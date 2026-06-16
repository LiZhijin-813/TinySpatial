"""Stage 1 训练脚本：ACM-MIM 自监督预训练。

执行流程：
    1. 加载多模态数据集（仅使用 BUS + SWE）
    2. 初始化 ACMMIMPretrainModel（TinyUSFM 完全可训练）
    3. AdamW 优化器 + 余弦退火学习率调度
    4. 训练/验证循环，early stopping，最佳模型保存

用法（在 TinySpatial_Project/ 目录下执行）：
    python code/train/train_stage1.py --pretrained_path TinyUSFM.pth
"""
import os
import sys
import argparse
import time
import datetime

# 项目根目录：确保 sys.path[0] 指向 TinySpatial_Project/ 而非脚本目录
# 避免 Python 内置 code 模块与项目的 code 包命名冲突
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 若 Python 已缓存内置 code 模块，则清除缓存以强制重新解析
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.utils.seed import seed_everything
from code.datasets.dataset import MultiModalBreastDataset
from code.models.stage1.pretrain_model import ACMMIMPretrainModel


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """执行一个 epoch 的训练。

    Args:
        model: ACMMIMPretrainModel 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备
        epoch: 当前 epoch 编号

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        bus_img = batch["bus_img"].to(device)
        swe_img = batch["swe_img"].to(device)

        # 前传：输出 (loss, pred_pixels, mask)
        loss, _, _ = model(bus_img, swe_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """在验证集上计算平均重建损失。

    Args:
        model: ACMMIMPretrainModel 模型
        dataloader: 验证数据加载器
        device: 计算设备

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        bus_img = batch["bus_img"].to(device)
        swe_img = batch["swe_img"].to(device)

        loss, _, _ = model(bus_img, swe_img)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main(args):
    """Stage 1 训练主函数。"""
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 输出目录：以时间戳命名
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "checkpoints", f"stage1_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    train_ds = MultiModalBreastDataset(
        root_dir=PROJECT_ROOT, split="train", img_size=args.img_size, max_text_len=args.max_text_len,
    )
    val_ds = MultiModalBreastDataset(
        root_dir=PROJECT_ROOT, split="val", img_size=args.img_size, max_text_len=args.max_text_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 初始化模型
    model = ACMMIMPretrainModel(
        pretrained_path=args.pretrained_path,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        img_size=args.img_size,
        mask_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # 优化器与学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: 连续 {args.patience} 个 epoch 无改善，于 epoch {epoch} 停止")
                break

    # 保存最终模型
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, os.path.join(output_dir, "final_model.pth"))
    print(f"训练完成。最佳 val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 ACM-MIM 自监督预训练")
    parser.add_argument("--pretrained_path", type=str, default=os.path.join(PROJECT_ROOT, "TinyUSFM.pth"),
                        help="TinyUSFM 预训练权重路径")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch 尺寸")
    parser.add_argument("--embed_dim", type=int, default=192, help="嵌入维度")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="BUS 掩码比例")
    parser.add_argument("--decoder_dim", type=int, default=96, help="解码器维度")
    parser.add_argument("--decoder_depth", type=int, default=2, help="解码器层数")
    parser.add_argument("--decoder_num_heads", type=int, default=6, help="解码器注意力头数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="学习率")
    parser.add_argument("--wd", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping 耐心值")
    parser.add_argument("--max_text_len", type=int, default=128, help="文本最大长度")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    args = parser.parse_args()
    main(args)
