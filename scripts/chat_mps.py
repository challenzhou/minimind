"""
MPS-aware interactive chat script for MiniMind on Apple Silicon.
Supports two formats:
  1. minimind-3 (Qwen3ForCausalLM, HuggingFace safetensors) — base model
  2. MiniMind (MiniMindForCausalLM, .pth) — custom trained weights

Usage:
    # minimind-3 base model (Qwen3, no fine-tuning):
    python scripts/chat_mps.py --model minimind3

    # Pretrain weights trained from scratch (MiniMind, .pth):
    python scripts/chat_mps.py --model minimind --weight notes_pretrain

    # Full SFT weights if trained (MiniMind, .pth):
    python scripts/chat_mps.py --model minimind --weight full_sft --mode sft
"""
import os
import sys
import time
import argparse
import warnings
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

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


def load_minimind3(args):
    """Load minimind-3 (Qwen3-based) from HuggingFace format, optionally with LoRA."""
    device = get_device()
    print(f"Device: {device}")

    minimind3_path = os.path.join(args.base_dir, 'out/gongjy/minimind-3')
    print(f"Loading minimind-3 from {minimind3_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(minimind3_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        minimind3_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if args.lora_weight and args.lora_weight != 'None':
        lora_path = os.path.join(args.base_dir, 'out', f"{args.lora_weight}.pth")
        print(f"Loading LoRA from {lora_path} ...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj'],
            bias='none', task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)
        model.load_state_dict(torch.load(lora_path, map_location='cpu'), strict=False)
        model = model.merge_and_unload()
        print("LoRA merged into base model")

    model = model.to(device).eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {total:.2f}M params ({total/1000:.2f}B)")
    return model, tokenizer, device


def load_minimind(args):
    """Load MiniMind (custom trained) from .pth weights."""
    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.base_dir, 'model'),
        trust_remote_code=True
    )

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    model = MiniMindForCausalLM(lm_config)

    weight_path = os.path.join(args.base_dir, f"out/{args.weight}.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found: {weight_path}")
    print(f"Loading weights from {weight_path} ...")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    if args.lora_weight and args.lora_weight != 'None':
        print(f"Applying LoRA: {args.lora_weight}")
        apply_lora(model)
        lora_path = os.path.join(args.base_dir, f"out/{args.lora_weight}.pth")
        load_lora(model, lora_path)

    model = model.half().eval().to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {total:.2f}M params")
    return model, tokenizer, device


def chat_loop(model, tokenizer, device, args):
    print("\n" + "=" * 60)
    print("MiniMind Chat (Apple Silicon MPS)")
    print("Commands: /reset, /quit, /context")
    print("=" * 60 + "\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

        # Build prompt
        if args.model == 'minimind3':
            # Qwen3 uses chat template
            prompt_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # MiniMind custom — raw completion or chat template
            if args.mode == 'sft':
                prompt_text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt_text = (tokenizer.bos_token or '') + user_input

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(device)

        print("\u1d4d\u200d: ", end='', flush=True)

        torch.manual_seed(random.randint(0, 31415926))
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
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        elapsed = time.time() - st
        speed = gen_tokens / elapsed if elapsed > 0 else 0
        print(f"\n\n[Speed: {speed:.1f} tokens/s, Generated: {gen_tokens} tokens]")

        conversation.append({"role": "assistant", "content": response})

        if device == 'mps':
            mem_allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"[MPS memory: {mem_allocated:.2f} GB]")


def main():
    parser = argparse.ArgumentParser(description="MiniMind MPS Chat")
    parser.add_argument('--model', default='minimind3',
                        choices=['minimind3', 'minimind'],
                        help="Model format: minimind3=Qwen3(HF), minimind=MiniMind(.pth)")
    parser.add_argument('--weight', default='notes_pretrain', type=str,
                        help="Weight name for minimind model")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Tokenizer path (defaults to model path)")
    parser.add_argument('--mode', default='sft', choices=['pretrain', 'sft'],
                        help="Generation mode (only for minimind)")
    parser.add_argument('--lora_weight', default=None, type=str,
                        help="LoRA weight name (only for minimind)")
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=4, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    args = parser.parse_args()

    # Resolve base_dir (parent of minimind/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minimind_root = os.path.dirname(script_dir)  # minimind/
    args.base_dir = minimind_root

    if args.model == 'minimind3':
        model, tokenizer, device = load_minimind3(args)
    else:
        model, tokenizer, device = load_minimind(args)

    chat_loop(model, tokenizer, device, args)


if __name__ == '__main__':
    main()
