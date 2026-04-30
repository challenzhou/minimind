"""
MPS-aware unified training script for MiniMind on Apple Silicon.
Supports: Pretrain, Full SFT, LoRA fine-tuning on MPS (Apple GPU).

Usage:
    # Convert notes to pretrain format
    python scripts/convert_notes_to_jsonl.py --notes_dir /path/to/notes --output dataset/notes_pretrain.jsonl

    # Pretrain on your notes (next-token prediction)
    python scripts/train_mps.py --mode pretrain --data_path dataset/notes_pretrain.jsonl --epochs 1 --batch_size 4

    # SFT on your notes (instruction tuning, jsonl with conversations)
    python scripts/train_mps.py --mode sft --data_path dataset/notes_sft.jsonl --epochs 3 --batch_size 2

    # LoRA fine-tune (recommended - fastest)
    python scripts/train_mps.py --mode lora --data_path dataset/notes_sft.jsonl --epochs 5 --batch_size 4
"""
import os
import sys
import json
import math
import random
import argparse

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_lora import apply_lora, save_lora
from transformers import AutoTokenizer


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def logger(content):
    if is_main_process():
        print(content)


def get_lr(step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


def init_mps_mode():
    """Initialize distributed mode (no-op for single-device MPS)."""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.mps.set_device(local_rank)
    return local_rank


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


def build_model(lm_config, from_weight, save_dir, device):
    """Load tokenizer and model, optionally load pretrained weights."""

    # Try minimind3 tokenizer first, then fallback to built-in model tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer_minimind3')
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(os.path.dirname(__file__), '../model')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = MiniMindForCausalLM(lm_config).to(device)

    if from_weight and from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            logger(f"Loading weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location=device)
            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logger(f"  Missing keys: {result.missing_keys[:5]}...")
            if result.unexpected_keys:
                logger(f"  Unexpected keys: {result.unexpected_keys[:5]}...")
        else:
            logger(f"[WARN] Weight file not found: {weight_path}, training from scratch")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger(f"Model params: {total:.2f}M")
    return model, tokenizer


class SkipBatchSampler:
    """Skip first N batches (for resume)."""
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)


def train_pretrain_epoch(model, loader, optimizer, scaler, autocast_ctx, args, lm_config, iters, start_step=0, epoch=0):
    model.train()
    step = start_step
    total_loss = 0

    for batch in loader:
        input_ids, labels = batch
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            outputs = model(input_ids, labels=labels)
            loss = (outputs.loss + outputs.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.log_interval == 0 and is_main_process():
            loss_val = loss.item() * args.accumulation_steps
            logger(f"Step {step+1}/{iters} | Loss: {loss_val:.4f} | LR: {lr:.6f}")

        step += 1

    return step


def train_sft_epoch(model, loader, optimizer, scaler, autocast_ctx, args, lm_config, iters, start_step=0, epoch=0, lora_params=None):
    model.train()
    step = start_step

    for batch in loader:
        input_ids, labels = batch
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            outputs = model(input_ids, labels=labels)
            loss = (outputs.loss + outputs.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        params_to_clip = lora_params if lora_params else model.parameters()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % args.log_interval == 0 and is_main_process():
            loss_val = loss.item() * args.accumulation_steps
            aux_val = outputs.aux_loss.item() if outputs.aux_loss else 0.0
            logger(f"Step {step+1}/{iters} | Loss: {loss_val:.4f} | Aux: {aux_val:.4f} | LR: {lr:.6f}")

        step += 1

    return step


def save_checkpoint(model, optimizer, scaler, epoch, step, lm_config, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    ckpt = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
        'lm_config': lm_config.__dict__,
    }
    torch.save(ckpt, save_path)
    logger(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MiniMind MPS Training")
    parser.add_argument("--mode", type=str, default="lora",
                        choices=["pretrain", "sft", "lora"],
                        help="Training mode")
    parser.add_argument("--save_dir", type=str, default="../out",
                        help="Model/checkpoint output directory")
    parser.add_argument("--save_name", type=str, default="notes_ft",
                        help="Checkpoint prefix name")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data jsonl")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per step")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Checkpoint save interval (steps)")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Model hidden dimension")
    parser.add_argument("--num_hidden_layers", type=int, default=8,
                        help="Number of transformer layers")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                        help="Use MoE architecture")
    parser.add_argument("--from_weight", type=str, default="none",
                        help="Base weight name prefix to load from save_dir (e.g. 'pretrain', 'full_sft')")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="MPS AMP dtype")
    parser.add_argument("--resume", type=int, default=0, choices=[0, 1],
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Resolve paths relative to minimind root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minimind_root = os.path.dirname(script_dir)  # goes from scripts/ -> minimind/
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(minimind_root, args.save_dir)
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(minimind_root, args.data_path)
    os.makedirs(save_dir, exist_ok=True)

    # Device
    args.device = "mps" if torch.mps.is_available() else "cpu"
    logger(f"Device: {args.device}")

    setup_seed(args.seed + 42)

    # Config
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    # AMP context for MPS
    if args.device == "mps":
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        autocast_ctx = autocast(device_type='mps', dtype=dtype)
        scaler = GradScaler('mps', enabled=True)
    else:
        autocast_ctx = nullcontext()
        scaler = None

    # Build model
    model, tokenizer = build_model(lm_config, args.from_weight, save_dir, args.device)

    # LoRA mode: apply LoRA adapters and freeze non-LoRA params
    lora_params = None
    if args.mode == "lora":
        apply_lora(model)
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
        total_p = sum(p.numel() for p in model.parameters()) / 1e6
        lora_p = sum(p.numel() for p in lora_params) / 1e6
        logger(f"LoRA params: {lora_p:.2f}M / {total_p:.2f}M total")
        # Freeze non-LoRA params
        for n, p in model.named_parameters():
            if 'lora' not in n:
                p.requires_grad = False
        optimizer_params = lora_params
    else:
        optimizer_params = model.parameters()

    # Dataset
    if args.mode == "pretrain":
        train_ds = PretrainDataset(data_path, tokenizer, max_length=args.max_seq_len)
    else:  # sft or lora
        train_ds = SFTDataset(data_path, tokenizer, max_length=args.max_seq_len)
    logger(f"Dataset size: {len(train_ds)} samples")

    # Optimizer
    optimizer = optim.AdamW(optimizer_params, lr=args.learning_rate)

    # Resume
    start_epoch = 0
    start_step = 0
    ckpt_path = os.path.join(save_dir, f"{args.save_name}.pth")
    if args.resume and os.path.exists(ckpt_path):
        logger(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        start_step = ckpt.get('step', 0)
        logger(f"Resumed: epoch={start_epoch}, step={start_step}")

    # Training loop
    total_steps = len(train_ds) // args.batch_size * args.epochs

    for epoch in range(start_epoch, args.epochs):
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                          num_workers=0, pin_memory=False)

        iters = len(loader) + skip

        if args.mode == "pretrain":
            train_pretrain_epoch(model, loader, optimizer, scaler, autocast_ctx,
                               args, lm_config, iters, start_step if epoch == start_epoch else 0, epoch)
        else:
            train_sft_epoch(model, loader, optimizer, scaler, autocast_ctx,
                          args, lm_config, iters, start_step if epoch == start_epoch else 0,
                          epoch, lora_params)

        start_step = 0  # After first epoch, start from 0

        # Save checkpoint after each epoch
        if is_main_process():
            save_checkpoint(model, optimizer, scaler, epoch + 1, 0,
                          lm_config, ckpt_path)

    logger("Training complete!")


if __name__ == '__main__':
    main()
