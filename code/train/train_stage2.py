"""Stage 2 训练脚本：SEC-Subtyping 微调训练。

保守微调策略（防止特征坍缩）：
    - 冻结编码器大部分层，仅解冻最后 2 层 + norm + patch_embed
    - 新组件（head/cross_attn/cdfi/ip_adapter）较高学习率，编码器低学习率
    - CE + 类别逆频率权重 + 轻度 label_smoothing，对抗类别不平衡
    - 特征多样性正则：惩罚 F_bus_swe 跨样本方差过小，强制编码器产出差异化特征
    - 梯度累积模拟更大 batch size，稳定梯度估计
    - 先关闭 CAM 对齐损失（beta=0），待分类有效后再开启
"""
import os
import sys
import argparse
import time
import datetime
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.utils.seed import seed_everything
from code.datasets.dataset import MultiModalBreastDataset
from code.models.stage2.subtyping_model import SECSubtypingModel
from code.models.stage2.cam import ContrastiveAlignmentModule


def compute_class_weights(dataset, num_classes=4, device="cpu"):
    """根据训练集类别逆频率计算 CrossEntropyLoss 的权重向量。"""
    from collections import Counter
    labels = [s["subtype_label"] for s in dataset.samples]
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        freq = counts.get(c, 1) / total
        weights[c] = 1.0 / (freq * num_classes)  # 逆频率归一化，使均值为1
    return weights


def build_optimizer(model, cam, base_lr, weight_decay):
    """构建 AdamW 优化器，分层学习率衰减。"""
    param_groups = []
    depth = len(model.encoder.blocks)
    unfreeze_last_n = 0
    # 检测实际解冻了多少层
    for i in range(depth):
        if any(p.requires_grad for p in model.encoder.blocks[i].parameters()):
            unfreeze_last_n = depth - i
            break

    # 新组件: 适中的学习率（避免 head 先于编码器收敛到退化解）
    param_groups.append({
        "params": model.mlp_head.parameters(),
        "lr": base_lr * 3,
        "weight_decay": 0.01,
        "name": "mlp_head",
    })
    param_groups.append({
        "params": model.ip_adapters.parameters(),
        "lr": base_lr * 2,
        "weight_decay": weight_decay,
        "name": "ip_adapter",
    })
    param_groups.append({
        "params": model.cross_attention.parameters(),
        "lr": base_lr * 2,
        "weight_decay": weight_decay,
        "name": "cross_attention",
    })
    param_groups.append({
        "params": model.cdfi_branch.parameters(),
        "lr": base_lr * 2,
        "weight_decay": weight_decay,
        "name": "cdfi_branch",
    })

    # 编码器: 仅解冻层使用低学习率，冻结层跳过
    for i, blk in enumerate(model.encoder.blocks):
        trainable_params = [p for p in blk.parameters() if p.requires_grad]
        if trainable_params:
            param_groups.append({
                "params": trainable_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": f"encoder_block_{i}",
            })

    # Patch embed + norm
    enc_params = [p for p in (
        list(model.encoder.patch_embed.parameters()) + list(model.encoder.norm.parameters())
    ) if p.requires_grad]
    if enc_params:
        param_groups.append({
            "params": enc_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "patch_embed_norm",
        })

    # 文本分支投影层（BERT 冻结，仅投影层可训练）
    text_proj_params = [p for n, p in model.text_branch.named_parameters() if p.requires_grad]
    if text_proj_params:
        param_groups.append({
            "params": text_proj_params,
            "lr": base_lr * 2,
            "weight_decay": weight_decay,
            "name": "text_proj",
        })

    # CAM
    param_groups.append({
        "params": cam.parameters(),
        "lr": base_lr * 2,
        "weight_decay": weight_decay,
        "name": "cam",
    })

    return torch.optim.AdamW(param_groups)


def train_one_epoch(model, cam, dataloader, optimizer, cls_criterion, device, epoch,
                    alpha, beta, gamma, max_grad_norm, accum_steps):
    """执行一个 epoch 的训练。

    Args:
        gamma: 特征多样性正则权重。当 F_bus_swe 跨 batch 方差低于阈值时施加惩罚。
        accum_steps: 梯度累积步数，模拟更大的 effective batch size。
    """
    model.train()
    cam.train()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    align_loss_sum = 0.0
    diversity_loss_sum = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        bus_img = batch["bus_img"].to(device)
        swe_img = batch["swe_img"].to(device)
        cdfi_img = batch["cdfi_img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["subtype_label"].to(device)

        logits, F_bus_swe, F_text = model(bus_img, swe_img, cdfi_img, input_ids, attention_mask)

        cls_loss = cls_criterion(logits, labels)
        if beta > 0:
            align_loss, _, _ = cam(F_bus_swe, F_text)
        else:
            align_loss = torch.tensor(0.0, device=device)

        # 特征多样性正则：惩罚 F_bus_swe 跨样本方差过小
        # 方差越大说明特征越有区分度，我们希望方差不低于 min_var
        if gamma > 0 and F_bus_swe.shape[0] > 1:
            feat_var = F_bus_swe.var(dim=0).mean()
            min_var = 0.01  # 期望的特征方差下界
            diversity_loss = F.relu(min_var - feat_var)  # 仅在方差不足时惩罚
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        loss = alpha * cls_loss + beta * align_loss + gamma * diversity_loss
        loss = loss / accum_steps  # 缩放以配合梯度累积

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(cam.parameters()),
                    max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

        total_loss_sum += loss.item() * accum_steps
        cls_loss_sum += cls_loss.item()
        align_loss_sum += align_loss.item()
        diversity_loss_sum += diversity_loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.shape[0]

        pbar.set_postfix(
            loss=f"{cls_loss.item():.4f}",
            div=f"{diversity_loss.item():.4f}",
            acc=f"{correct/total:.3f}",
        )

    return {
        "total_loss": total_loss_sum / len(dataloader),
        "cls_loss": cls_loss_sum / len(dataloader),
        "align_loss": align_loss_sum / len(dataloader),
        "diversity_loss": diversity_loss_sum / len(dataloader),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def validate(model, cam, dataloader, cls_criterion, device, alpha, beta, num_classes=4):
    """在验证集上评估模型性能。"""
    model.eval()
    cam.eval()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    all_feat_var = []

    for batch in dataloader:
        bus_img = batch["bus_img"].to(device)
        swe_img = batch["swe_img"].to(device)
        cdfi_img = batch["cdfi_img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["subtype_label"].to(device)

        logits, F_bus_swe, F_text = model(bus_img, swe_img, cdfi_img, input_ids, attention_mask)
        cls_loss = cls_criterion(logits, labels)
        if beta > 0:
            align_loss, _, _ = cam(F_bus_swe, F_text)
        else:
            align_loss = torch.tensor(0.0, device=device)

        loss = alpha * cls_loss + beta * align_loss
        total_loss_sum += loss.item()
        cls_loss_sum += cls_loss.item()

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())
        all_feat_var.append(F_bus_swe.var(dim=0).mean().item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    pred_dist = np.bincount(all_preds, minlength=num_classes)
    label_dist = np.bincount(all_labels, minlength=num_classes)

    # 特征坍缩检测：logits 方差和 F_bus_swe 方差
    all_logits = torch.cat(all_logits, dim=0)
    logits_var = all_logits.var(dim=0).mean().item()
    feat_var = np.mean(all_feat_var)

    return {
        "total_loss": total_loss_sum / len(dataloader),
        "cls_loss": cls_loss_sum / len(dataloader),
        "acc": acc,
        "preds": all_preds,
        "labels": all_labels,
        "pred_dist": pred_dist.tolist(),
        "label_dist": label_dist.tolist(),
        "logits_var": logits_var,
        "feat_var": feat_var,
    }


class LinearWarmupCosineScheduler:
    """线性 warmup + 余弦退火学习率调度器。"""

    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def main(args):
    """Stage 2 端到端训练主函数。"""
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "checkpoints", f"stage2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    train_ds = MultiModalBreastDataset(
        root_dir=PROJECT_ROOT, split="train", img_size=args.img_size, max_text_len=args.max_text_len,
        metadata_file=args.metadata_file,
    )
    val_ds = MultiModalBreastDataset(
        root_dir=PROJECT_ROOT, split="val", img_size=args.img_size, max_text_len=args.max_text_len,
        metadata_file=args.metadata_file,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 初始化模型 — 保守微调：仅解冻编码器最后 2 层
    model = SECSubtypingModel(
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes,
        img_size=args.img_size,
        unfreeze_last_n=args.unfreeze,
    ).to(device)

    # CAM 对比对齐模块
    cam = ContrastiveAlignmentModule(
        image_dim=192, text_dim=512, proj_dim=512, temperature=args.tau,
    ).to(device)

    # CE + 类别逆频率权重 + 轻度 label_smoothing，对抗类别不平衡
    class_weights = compute_class_weights(train_ds, num_classes=args.num_classes, device=device)
    print(f"类别权重: {class_weights.tolist()}")
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_p/1e6:.2f}M, 可训练: {trainable/1e6:.2f}M")
    print(f"学习率: {args.lr}, warmup: {args.warmup}, beta: {args.beta}, gamma: {args.gamma}")
    print(f"解冻编码器层数: {args.unfreeze}, 梯度累积: {args.accum_steps} (effective_bs={args.batch_size * args.accum_steps})")

    optimizer = build_optimizer(model, cam, args.lr, args.wd)
    scheduler = LinearWarmupCosineScheduler(optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        scheduler.step(epoch - 1)

        train_metrics = train_one_epoch(
            model, cam, train_loader, optimizer, cls_criterion, device, epoch,
            args.alpha, args.beta, args.gamma, args.max_grad_norm, args.accum_steps,
        )
        val_metrics = validate(
            model, cam, val_loader, cls_criterion, device, args.alpha, args.beta,
            num_classes=args.num_classes,
        )
        elapsed = time.time() - t0

        lrs = scheduler.get_last_lr()
        lr_str = f"lr={lrs[0]:.2e}"

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['cls_loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
            f"val_loss={val_metrics['cls_loss']:.4f} val_acc={val_metrics['acc']:.3f} | "
            f"pred={val_metrics['pred_dist']} | "
            f"logits_var={val_metrics['logits_var']:.4f} feat_var={val_metrics['feat_var']:.6f} | "
            f"div_loss={train_metrics['diversity_loss']:.4f} | "
            f"{lr_str} {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k not in ("preds", "labels")},
            "lr": lrs,
        })

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cam_state_dict": cam.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["acc"],
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  -> 保存最佳模型 (val_acc={val_metrics['acc']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: 连续 {args.patience} 个 epoch 无改善，于 epoch {epoch} 停止")
                break

    torch.save({
        "model_state_dict": model.state_dict(),
        "cam_state_dict": cam.state_dict(),
    }, os.path.join(output_dir, "final_model.pth"))

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n训练完成。最佳 val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 SEC-Subtyping 端到端训练")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Stage 1 ACM-MIM 预训练权重路径")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv",
                        help="元数据文件名 (4分类用 metadata.csv, 5分类用 metadata_5class.csv)")
    parser.add_argument("--num_classes", type=int, default=4, help="分类类别数 (4 或 5)")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="基础学习率")
    parser.add_argument("--wd", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--warmup", type=int, default=5, help="线性 warmup epoch 数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--alpha", type=float, default=1.0, help="L_cls 权重系数")
    parser.add_argument("--beta", type=float, default=0.0, help="L_align 权重系数 (0=关闭)")
    parser.add_argument("--gamma", type=float, default=1.0, help="特征多样性正则权重 (0=关闭)")
    parser.add_argument("--tau", type=float, default=0.07, help="CAM 温度参数 τ")
    parser.add_argument("--unfreeze", type=int, default=2, help="解冻编码器最后 N 层")
    parser.add_argument("--accum_steps", type=int, default=4, help="梯度累积步数 (effective_bs=batch_size*accum_steps)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping 耐心值")
    parser.add_argument("--max_text_len", type=int, default=128, help="文本最大长度")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    args = parser.parse_args()
    main(args)
