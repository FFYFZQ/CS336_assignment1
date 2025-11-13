"""
训练循环脚本

这个脚本实现了一个完整的训练循环来训练Transformer语言模型，包括：
1. 命令行参数配置模型和优化器超参数
2. 使用np.memmap高效加载训练和验证数据集
3. 保存和加载检查点
4. 定期记录训练和验证性能
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path
import wandb
from tqdm import tqdm
import json

from .model import Transformer
from .AdamW import AdamW
from .utils import (
    get_batch_data,
    cross_entropy,
    cosine_annealing,
    grad_clip,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练Transformer语言模型")

    # 模型超参数
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1024, help="前馈网络维度")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度")
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="RoPE theta参数"
    )

    # 优化器超参数
    parser.add_argument("--lr_max", type=float, default=6e-4, help="最大学习率")
    parser.add_argument("--lr_min", type=float, default=6e-5, help="最小学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--warmup_iters", type=int, default=100, help="预热迭代次数")
    parser.add_argument("--max_iters", type=int, default=5000, help="最大训练迭代次数")
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="梯度裁剪的最大范数"
    )

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="训练设备",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型",
    )

    # 数据路径
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="训练数据路径（.npy或.bin文件）",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="验证数据路径（.npy或.bin文件）",
    )

    # 检查点
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500, help="检查点保存间隔"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="从检查点恢复训练"
    )

    # 评估和日志
    parser.add_argument("--eval_interval", type=int, default=100, help="评估间隔")
    parser.add_argument("--eval_iters", type=int, default=20, help="评估时的迭代次数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument(
        "--use_wandb", action="store_true", help="是否使用Weights & Biases记录"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="transformer-lm", help="W&B项目名称"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B运行名称")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dtype=np.int32):
    """
    使用np.memmap加载大型数据集

    Args:
        data_path: 数据文件路径
        dtype: 数据类型

    Returns:
        memmap数组
    """
    if data_path.endswith(".npy"):
        # 如果是.npy文件，使用np.load加载
        data = np.load(data_path, mmap_mode="r")
    elif data_path.endswith(".bin"):
        # 如果是.bin文件，使用memmap直接加载
        data = np.memmap(data_path, dtype=dtype, mode="r")
    else:
        raise ValueError(f"不支持的文件格式: {data_path}，请使用.npy或.bin文件")

    return data


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters, batch_size, context_length, device):
    """
    估算模型在数据集上的平均损失

    Args:
        model: 模型
        dataset: 数据集
        eval_iters: 评估迭代次数
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备

    Returns:
        平均损失
    """
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch_data(dataset, batch_size, context_length, device)
        logits = model(x)  # (batch_size, seq_len, vocab_size)

        # 计算交叉熵损失
        # 需要将logits和targets reshape以适配cross_entropy函数
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = y.reshape(batch_size * seq_len)

        loss = cross_entropy(logits_flat, targets_flat)
        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def train(args):
    """主训练函数"""

    # 设置随机种子
    set_seed(args.seed)

    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # 初始化W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

    # 加载数据集
    print("加载数据集...")
    train_data = load_dataset(args.train_data_path)
    val_data = load_dataset(args.val_data_path)
    print(f"训练数据大小: {len(train_data):,} tokens")
    print(f"验证数据大小: {len(val_data):,} tokens")

    # 初始化模型
    print("初始化模型...")
    model = Transformer(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=dtype,
    )
    model = model.to(args.device)

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,  # 初始学习率，后面会动态调整
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 从检查点恢复
    start_iter = 0
    if args.resume_from is not None:
        print(f"从检查点恢复: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"从迭代 {start_iter} 恢复训练")

    # 训练循环
    print(f"\n开始训练，从迭代 {start_iter} 到 {args.max_iters}...")
    model.train()

    for iteration in tqdm(
        range(start_iter, args.max_iters), initial=start_iter, total=args.max_iters
    ):
        # 更新学习率（cosine annealing with warmup）
        lr = cosine_annealing(
            t=iteration,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            t_w=args.warmup_iters,
            t_c=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 获取训练批次
        x, y = get_batch_data(
            train_data, args.batch_size, args.context_length, args.device
        )

        # 前向传播
        logits = model(x)  # (batch_size, seq_len, vocab_size)

        # 计算损失
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = y.reshape(batch_size * seq_len)
        loss = cross_entropy(logits_flat, targets_flat)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.grad_clip_norm > 0:
            grad_clip(model.parameters(), args.grad_clip_norm)

        # 更新参数
        optimizer.step()

        # 日志记录
        if iteration % args.log_interval == 0 or iteration == args.max_iters - 1:
            print(f"\n[Iter {iteration}] train_loss: {loss.item():.4f}, lr: {lr:.6f}")

            if args.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/iteration": iteration,
                    }
                )

        # 评估
        if iteration % args.eval_interval == 0 or iteration == args.max_iters - 1:
            print(f"\n评估 (迭代 {iteration})...")
            val_loss = estimate_loss(
                model,
                val_data,
                args.eval_iters,
                args.batch_size,
                args.context_length,
                args.device,
            )
            print(f"[Iter {iteration}] val_loss: {val_loss:.4f}")

            if args.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/iteration": iteration,
                    }
                )

        # 保存检查点
        if iteration % args.checkpoint_interval == 0 or iteration == args.max_iters - 1:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
            print(f"\n保存检查点到 {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

    print("\n训练完成！")

    if args.use_wandb:
        wandb.finish()


def main():
    """主函数"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
