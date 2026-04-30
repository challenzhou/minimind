"""
Convert minimind-3 (Qwen3ForCausalLM, safetensors) to MiniMind format (.pth)
for continued pretraining/fine-tuning.

This allows you to:
  1. Start from minimind-3's pretrained weights (Qwen3, 768-dim, 8 layers)
  2. Convert to MiniMindForCausalLM format
  3. Continue pretraining or SFT on your notes

Usage:
    python scripts/convert_minimind3.py
"""
import os
import sys
import argparse
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


def qwen3_to_minimind(state_dict, src_config):
    """
    Map Qwen3 state dict keys to MiniMind state dict keys.
    The architectures are very similar (RoPE, RMSNorm, Gated SiLU).
    """
    dst_dict = {}
    for k, v in state_dict.items():
        # Model prefix
        if k.startswith('model.'):
            new_k = k.replace('model.', 'model.')
        elif k.startswith('lm_head.'):
            new_k = k
        else:
            new_k = k

        # Layer renaming: model.layers.N.* -> model.blocks.N.*
        if '.layers.' in new_k:
            new_k = new_k.replace('.layers.', '.blocks.')

        # Embedding
        if new_k == 'model.embed_tokens.weight':
            new_k = 'model.tok_embeddings.weight'

        dst_dict[new_k] = v

    return dst_dict


def convert_minimind3_to_minimind(args):
    """Convert minimind-3 safetensors to MiniMind .pth for continued training."""

    minimind3_path = os.path.join(args.minimind3_dir, 'out/gongjy/minimind-3')

    print(f"Loading Qwen3ForCausalLM from {minimind3_path} ...")
    qwen3 = AutoModelForCausalLM.from_pretrained(
        minimind3_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    state_dict = qwen3.state_dict()
    src_config = qwen3.config

    print(f"Source config: hidden_size={src_config.hidden_size}, "
          f"num_layers={src_config.num_hidden_layers}, "
          f"vocab_size={src_config.vocab_size}")

    # Build MiniMind config matching the source
    lm_config = MiniMindConfig(
        hidden_size=src_config.hidden_size,
        num_hidden_layers=src_config.num_hidden_layers,
        vocab_size=src_config.vocab_size,
        intermediate_size=src_config.intermediate_size,
        num_attention_heads=src_config.num_attention_heads,
        num_key_value_heads=src_config.num_key_value_heads,
        max_seq_len=src_config.max_position_embeddings,
        rope_theta=src_config.rope_theta,
        use_moe=False,
    )

    print(f"Creating MiniMind model with hidden_size={lm_config.hidden_size}, "
          f"num_layers={lm_config.num_hidden_layers}")

    minimind = MiniMindForCausalLM(lm_config)

    # Map state dict
    mapped = qwen3_to_minimind(state_dict, src_config)
    minimind.load_state_dict(mapped, strict=False)

    total = sum(p.numel() for p in minimind.parameters()) / 1e6
    print(f"Converted model: {total:.2f}M params")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.pth")
    torch.save(minimind.state_dict(), out_path)
    print(f"Saved to {out_path}")

    # Also copy tokenizer files
    tokenizer_out = os.path.join(args.out_dir, 'tokenizer_minimind3')
    os.makedirs(tokenizer_out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(minimind3_path, trust_remote_code=True)
    tokenizer.save_pretrained(tokenizer_out)
    print(f"Tokenizer saved to {tokenizer_out}")

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimind3_dir', default='/Users/challenzhou/CodeGeeXProjects/minimind',
                        help="Root of minimind project")
    parser.add_argument('--out_dir', default='/Users/challenzhou/CodeGeeXProjects/minimind/out',
                        help="Output directory for .pth")
    parser.add_argument('--name', default='minimind3_pretrain',
                        help="Output weight name")
    args = parser.parse_args()

    convert_minimind3_to_minimind(args)
