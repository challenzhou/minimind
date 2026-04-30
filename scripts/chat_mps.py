"""
MPS-aware interactive chat script for MiniMind on Apple Silicon.
Works with both pretrain weights (raw completion) and SFT/LoRA weights (chat format).

Usage:
    # Pretrain mode (raw text completion):
    python scripts/chat_mps.py --weight notes_pretrain --mode pretrain

    # SFT/LoRA mode (chat format):
    python scripts/chat_mps.py --weight notes_sft --mode sft

    # With custom path:
    python scripts/chat_mps.py --weight full_sft --save_dir out --mode sft
"""
import os
import sys
import time
import argparse
import warnings
import random

import torch
from transformers import AutoTokenizer, TextStreamer

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


def init_model(args):
    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    model = MiniMindForCausalLM(lm_config)

    moe_suffix = '_moe' if args.use_moe else ''
    weight_path = f'{args.save_dir}/{args.weight}{moe_suffix}.pth'
    if not os.path.exists(weight_path):
        # Try without suffix
        weight_path = f'{args.save_dir}/{args.weight}.pth'

    print(f"Loading weights from {weight_path} ...")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()

    if args.lora_weight and args.lora_weight != 'None':
        print(f"Applying LoRA: {args.lora_weight}")
        apply_lora(model)
        lora_path = f'{args.save_dir}/{args.lora_weight}.pth'
        load_lora(model, lora_path)

    model = model.half().eval().to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {total:.2f}M params")
    return model, tokenizer, device


def chat_loop(model, tokenizer, device, args):
    """Interactive chat loop."""
    conversation = []

    print("\n" + "=" * 60)
    print("MiniMind Chat (Apple Silicon MPS)")
    print("Commands: /reset, /quit, /context")
    print("=" * 60 + "\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            user_input = input("\u4f60: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == '/quit':
            print("Goodbye!")
            break
        elif user_input == '/reset':
            conversation = []
            print("[Conversation reset]")
            continue
        elif user_input == '/context':
            print(f"[Conversation has {len(conversation)} messages]")
            for i, m in enumerate(conversation):
                print(f"  {i+1}. [{m['role']}]: {m['content'][:80]}...")
            continue

        conversation.append({"role": "user", "content": user_input})

        if args.mode == 'pretrain' or 'pretrain' in args.weight:
            # Pretrain mode: raw completion
            prompt = tokenizer.bos_token + user_input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        else:
            # SFT/LoRA mode: chat template
            prompt_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(device)

        print("\u1d4d\u200d: ", end='', flush=True)

        setup_seed(random.randint(0, 31415926))
        st = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        if not streamer:
            print(response, end='', flush=True)

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        elapsed = time.time() - st
        speed = gen_tokens / elapsed if elapsed > 0 else 0
        print(f"\n\n[Speed: {speed:.1f} tokens/s, Generated: {gen_tokens} tokens]")

        conversation.append({"role": "assistant", "content": response})

        # Memory check
        if device == 'mps':
            mem_allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"[MPS memory: {mem_allocated:.2f} GB]")


def main():
    parser = argparse.ArgumentParser(description="MiniMind MPS Chat")
    parser.add_argument('--weight', default='notes_pretrain', type=str,
                        help="Weight name prefix (without .pth)")
    parser.add_argument('--save_dir', default='../out', type=str,
                        help="Directory containing weights")
    parser.add_argument('--tokenizer_path', default='../model', type=str,
                        help="Tokenizer directory")
    parser.add_argument('--mode', default='sft', choices=['pretrain', 'sft'],
                        help="Generation mode: pretrain=raw completion, sft=chat")
    parser.add_argument('--lora_weight', default=None, type=str,
                        help="LoRA weight name (optional)")
    parser.add_argument('--hidden_size', default=512, type=int,
                        help="Model hidden dimension")
    parser.add_argument('--num_hidden_layers', default=4, type=int,
                        help="Number of layers")
    parser.add_argument('--use_moe', default=0, type=int,
                        help="Use MoE architecture")
    parser.add_argument('--max_new_tokens', default=512, type=int,
                        help="Max tokens to generate")
    parser.add_argument('--temperature', default=0.85, type=float,
                        help="Sampling temperature")
    parser.add_argument('--top_p', default=0.95, type=float,
                        help="Nucleus sampling threshold")
    args = parser.parse_args()

    # Resolve paths relative to script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minimind_root = os.path.dirname(script_dir)

    for arg_name in ['save_dir', 'tokenizer_path']:
        val = getattr(args, arg_name)
        if not os.path.isabs(val):
            setattr(args, arg_name, os.path.join(minimind_root, val))

    model, tokenizer, device = init_model(args)
    chat_loop(model, tokenizer, device, args)


if __name__ == '__main__':
    main()
