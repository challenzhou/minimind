#!/usr/bin/python3
"""
LoRA fine-tuning on Apple Silicon (MPS) using HuggingFace PEFT on minimind3 (Qwen3).

This script:
  1. Loads minimind3 as Qwen3ForCausalLM (HuggingFace safetensors, NOT MiniMind .pth)
  2. Applies LoRA adapters to Qwen3 layers
  3. Continues pretraining on your notes (notes_pretrain.jsonl)
  4. Saves LoRA adapters to out/lora_notes.pth

Inference: Load base minimind3 + LoRA adapters for chat.

Usage:
    python scripts/lora_mps.py --data_path dataset/notes_pretrain.jsonl --lora_rank 16 --epochs 1
"""
import os
import sys
import math
import time
import random
import argparse
import warnings

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


class PretrainDataset(torch.utils.data.Dataset):
    """Load jsonl pretrain data."""
    def __init__(self, path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                import json
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                    if text:
                        self.data.append(text)
                except:
                    pass
        print(f"Loaded {len(self.data)} documents")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc.input_ids.squeeze(0)
        labels = input_ids.clone()
        return input_ids, labels


def data_collate(batch, pad_id):
    """Collate with padding, create attention_mask."""
    max_len = max(len(b[0]) for b in batch)
    input_ids, labels = [], []
    for b in batch:
        pad_len = max_len - len(b[0])
        input_ids.append(torch.cat([b[0], torch.full((pad_len,), pad_id, dtype=b[0].dtype)]))
        labels.append(torch.cat([b[1], torch.full((pad_len,), -100, dtype=b[1].dtype)]))
    return torch.stack(input_ids), torch.stack(labels)


class SkipBatchSampler:
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch, skipped = [], 0
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


def get_lr(step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


def prune_checkpoints(save_name, save_dir, max_keep=3):
    """Remove all step checkpoints for save_name except the most recent max_keep."""
    import glob
    pattern = os.path.join(save_dir, f"{save_name}_step_*.pth")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    removed = 0
    for ckpt in checkpoints[max_keep:]:
        os.remove(ckpt)
        removed += 1
    if removed:
        print(f"Pruned {removed} old checkpoint(s) — kept {max_keep} most recent")


def train_epoch(model, loader, optimizer, scaler, autocast_ctx, args, iters, start_step=0, epoch=0):
    model.train()
    step = start_step
    total_loss = 0
    device = args.device

    for batch in loader:
        # batch is (input_ids, labels) from collate — unpack properly
        if isinstance(batch, (tuple, list)):
            input_ids_batch, labels_batch = batch[0].to(device), batch[1].to(device)
        else:
            input_ids_batch = batch.to(device)
            labels_batch = None

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(input_ids_batch, labels=labels_batch)
            loss = outputs.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * args.accumulation_steps

        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / (step + 1 - start_step)
            print(f"Step {step+1}/{iters} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | GPU mem: {torch.mps.current_allocated_memory()/1024**3:.2f}GB")

        if (step + 1) % args.save_interval == 0 and step + 1 != iters:
            save_path = os.path.join(args.save_dir, f"{args.save_name}_step_{step+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
            if args.max_keep > 0:
                prune_checkpoints(args.save_name, args.save_dir, args.max_keep)

        step += 1

    return total_loss / max(1, step - start_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='out')
    parser.add_argument('--save_name', type=str, default='lora_notes')
    parser.add_argument('--minimind3_dir', type=str, default=None,
                        help='Path to minimind project root (default: auto-detect)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--max_keep', type=int, default=3,
                        help='Max step checkpoints to keep (prunes oldest, default: 3, 0=keep all)')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Resolve paths
    if args.minimind3_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        minimind_root = os.path.dirname(script_dir)
        args.minimind3_dir = minimind_root
    args.save_dir = os.path.join(args.minimind3_dir, args.save_dir)
    args.data_path = os.path.join(args.minimind3_dir, args.data_path)
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    args.device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Device: {args.device}")
    setup_seed(args.seed)

    # AMP dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    autocast_ctx = autocast(device_type='mps', dtype=dtype)
    scaler = GradScaler('mps', enabled=True)

    # Load minimind3 as Qwen3ForCausalLM
    minimind3_path = os.path.join(args.minimind3_dir, 'out/gongjy/minimind-3')
    print(f"Loading minimind3 from {minimind3_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(minimind3_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        minimind3_path, trust_remote_code=True, torch_dtype=dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Apply LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(args.device)
    model.model.config.use_cache = False

    # Dataset
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"Dataset: {len(train_ds)} docs")

    # Optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Training
    total_steps = len(train_ds) // args.batch_size * args.epochs
    print(f"Total steps: {total_steps}")

    indices = list(range(len(train_ds)))
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    batch_sampler = SkipBatchSampler(sampler, args.batch_size)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0,
                        collate_fn=lambda b: data_collate(b, tokenizer.pad_token_id or 0))

    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, scaler, autocast_ctx, args,
                          len(loader), 0, epoch)
        print(f"Epoch {epoch} done | Avg loss: {loss:.4f}")

    # Save LoRA with verify
    out_path = os.path.join(args.save_dir, f"{args.save_name}.pth")
    torch.save(model.state_dict(), out_path)
    # Verify save
    verify = torch.load(out_path, map_location='cpu')
    print(f"LoRA saved to {out_path} ({len(verify)} keys verified)")
    print(f"Load for inference: base model + get_peft_model + load_state_dict(torch.load('{out_path}', map_location='cpu'))")
    print(f"\nInference example:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('./out/gongjy/minimind-3')")
    print(f"  model = get_peft_model(model, lora_config)")
    print(f"  model.load_state_dict(torch.load('{out_path}'))")


if __name__ == '__main__':
    main()
