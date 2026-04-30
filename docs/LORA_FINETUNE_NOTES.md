# Fine-tuning MiniMind on Your Notes (Apple Silicon MPS)

This guide walks through the complete pipeline to fine-tune **MiniMind-3** (63.9M params, Qwen3-based) on your personal notes using LoRA on an Apple Silicon Mac with MPS GPU — no NVIDIA GPU required.

---

## Pipeline Overview

```
1. Fork & clone MiniMind repo
2. Download minimind-3 base weights
3. Convert your notes → JSONL pretrain format
4. Install Python dependencies
5. Run LoRA fine-tuning (MPS)
6. Test inference with fine-tuned model
```

---

## Step 1: Fork and Clone the Repo

Fork [MiniMind](https://github.com/jingyaogong/minimind) on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/minimind.git
cd minimind
```

Add the original repo as `upstream` for future updates:

```bash
git remote add upstream https://github.com/jingyaogong/minimind.git
```

---

## Step 2: Download MiniMind-3 Base Weights

MiniMind-3 is the Qwen3-based model (63.9M params). The weights are hosted on ModelScope.

```bash
# Create the expected directory structure
mkdir -p out/gongjy

# Download via modelscope CLI (or manually from https://modelscope.cn/models/gongjy/minimind-3)
pip install modelscope
modelscope download --model_id gongjy/minimind-3 --save_dir out/gongjy/minimind-3
```

After download, verify the structure:

```
out/gongjy/minimind-3/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

> **Why minimind-3?** MiniMind-3 uses the Qwen3 architecture (HuggingFace safetensors format), which is the recommended starting point for fine-tuning. The older MiniMind-1/2 models use a custom architecture that has compatibility issues when converted back from Qwen3 checkpoints.

---

## Step 3: Convert Notes to JSONL Pretrain Format

Your notes (`.ipynb` or `.md` files) need to be converted into a JSONL file where each line is a JSON object with a `text` field.

Create a conversion script at `scripts/convert_notes_to_jsonl.py`:

```python
"""
Convert .ipynb / .md notes to minimind pretrain JSONL format.
Each line: {"text": "<notebook content as plain text>"}
"""
import json
import os
import re
from pathlib import Path


def extract_notebook_text(nb_path):
    """Extract text from a .ipynb Jupyter notebook."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    texts = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            texts.append(''.join(cell['source']))
        elif cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if src.strip():
                texts.append(f"```python\n{src}\n```")
    return '\n\n'.join(texts)


def extract_md_text(md_path):
    """Extract text from a .md file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Remove code block fences for cleaner pretrain text
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # remove links
    return content.strip()


def convert_notes(notes_dir, output_path, chunk_size=1000):
    """
    Walk notes_dir, convert all .ipynb and .md files to JSONL.
    Splits long documents into chunks of ~chunk_size characters.
    """
    notes_path = Path(notes_dir)
    output_file = open(output_path, 'w', encoding='utf-8')

    for fp in notes_path.rglob('*'):
        if fp.suffix not in ('.ipynb', '.md'):
            continue
        try:
            text = extract_notebook_text(fp) if fp.suffix == '.ipynb' else extract_md_text(fp)
            text = text.strip()
            if not text or len(text) < 50:
                continue
            # Split long documents into chunks to fit max_seq_len
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 50:
                    output_file.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    output_file.close()
    print(f"Converted notes saved to {output_path}")


if __name__ == '__main__':
    import sys
    notes_dir = sys.argv[1]  # e.g., /Users/challenzhou/CodeGeeXProjects/notes
    output_path = sys.argv[2]  # e.g., dataset/notes_pretrain.jsonl
    convert_notes(notes_dir, output_path)
```

Run it:

```bash
cd /Users/challenzhou/CodeGeeXProjects/minimind

python3 scripts/convert_notes_to_jsonl.py \
    /Users/challenzhou/CodeGeeXProjects/notes \
    dataset/notes_pretrain.jsonl
```

This produces `dataset/notes_pretrain.jsonl` with one JSON object per line:

```json
{"text": "Random matrix theory basics..."}
{"text": "## Chern-Simons Form\n\nThe action is..."}
```

---

## Step 4: Install Python Dependencies

```bash
cd /Users/challenzhou/CodeGeeXProjects/minimind

# Core dependencies
pip3 install torch transformers datasets peft sentencepiece

# Verify MPS is available (Apple Silicon)
python3 -c "import torch; print('MPS:', torch.mps.is_available()); print('Device:', 'mps' if torch.mps.is_available() else 'cpu')"
```

Expected output:

```
MPS: True
Device: mps
```

---

## Step 5: Run LoRA Fine-tuning on MPS

The training script is at `scripts/lora_mps.py`. Key configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| model | minimind-3 (Qwen3) | 63.9M params total |
| LoRA rank | 16 | 1.9M trainable params (2.9%) |
| LoRA alpha | 32 | Scale factor |
| LoRA dropout | 0.05 | |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers |
| batch size | 8 | Fits in 18GB Apple Silicon |
| max seq len | 512 | |
| learning rate | 3e-4 | PEFT standard LR |
| dtype | bfloat16 | MPS-compatible |

Launch training:

```bash
cd /Users/challenzhou/CodeGeeXProjects/minimind

PYTHONUNBUFFERED=1 python3 -u scripts/lora_mps.py \
    --data_path dataset/notes_pretrain.jsonl \
    --save_dir out \
    --save_name lora_notes \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --max_seq_len 512 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --dtype bfloat16 \
    --log_interval 50 \
    --save_interval 250 \
    2>&1 | tee /tmp/lora_train.log
```

**Expected output:**

```
Device: mps
Loading minimind3 from out/gongjy/minimind-3 ...
Model params: 63.91M
trainable params: 1,916,928 || all params: 65,829,120 || trainable%: 2.9120
Loaded 12164 documents
Dataset: 12164 docs
Total steps: 1520
Step 50/1521 | Loss: 2.98 | LR: 0.000299 | GPU mem: 0.22GB
Step 100/1521 | Loss: 2.79 | LR: 0.000297 | GPU mem: 0.22GB
...
Step 1500/1521 | Loss: 2.27 | LR: 0.000030 | GPU mem: 0.22GB
Checkpoint saved: out/lora_notes_step_1500.pth
Epoch 0 done | Avg loss: 2.2691
LoRA saved to out/lora_notes.pth
```

**Hardware requirements:**
- Apple Silicon Mac with 18GB+ unified memory
- ~40 minutes for 1 epoch on M3 Pro
- MPS memory usage: ~0.22GB (LoRA is very memory-efficient)

**Checkpoint locations:**

| File | Step | When saved |
|------|------|-----------|
| `out/lora_notes_step_250.pth` | 250 | first checkpoint |
| `out/lora_notes_step_500.pth` | 500 | second checkpoint |
| `out/lora_notes_step_1000.pth` | 1000 | third checkpoint |
| `out/lora_notes.pth` | 1520 | final merged |

To train for more epochs (recommended for better knowledge absorption):

```bash
python3 -u scripts/lora_mps.py \
    --data_path dataset/notes_pretrain.jsonl \
    --save_dir out \
    --save_name lora_notes_v2 \
    --epochs 3 \
    --save_interval 500 \
    # ... rest of args same as above
```

---

## Step 6: Test Inference with Fine-tuned Model

Load the base model, apply LoRA adapters, merge, and generate:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

minimind3_path = 'out/gongjy/minimind-3'

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(minimind3_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(minimind3_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA adapters
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    bias='none', task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load('out/lora_notes.pth', map_location='cpu'), strict=False)

# Merge LoRA weights into base for clean inference
model = model.merge_and_unload()
model.eval()

# Generate
inputs = tokenizer('Random matrix theory', return_tensors='pt')
with torch.no_grad():
    output = model.generate(inputs.input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Expected output (Chinese physics content from your notes):

```
Random matrix theory.

### 3. 量子几何与几何

#### 3.1 量子几何

- **量子几何**：量子态的几何结构
- **几何解释**：量子态的几何结构
```

---

## Project Structure

```
minimind/
├── out/
│   ├── gongjy/minimind-3/       # Base model (downloaded)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer_*
│   ├── lora_notes.pth           # Fine-tuned LoRA adapters (129MB)
│   ├── lora_notes_step_500.pth  # Mid-training checkpoint
│   └── ...
├── dataset/
│   └── notes_pretrain.jsonl     # Converted notes (12,164 chunks)
├── scripts/
│   ├── lora_mps.py               # LoRA fine-tuning script (MPS)
│   ├── chat_mps.py               # Interactive chat script
│   └── convert_notes_to_jsonl.py # Notes → JSONL converter
└── model/
    └── model_minimind.py         # MiniMind architecture (PyTorch native)
```

---

## Troubleshooting

**MPS not available:**
```python
import torch
print(torch.mps.is_available())  # Must be True on Apple Silicon
```
If False, update to the latest macOS + PyTorch nightly: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu`

**Loss not decreasing:**
- Try lower learning rate (1e-4 instead of 3e-4)
- Check that notes_pretrain.jsonl actually contains text: `head -1 dataset/notes_pretrain.jsonl`

**Garbled output / repetitive tokens:**
- This is expected with the 63.9M base model on very specialized topics without fine-tuning
- LoRA fine-tuning should fix this — verify `lora_notes.pth` was loaded correctly
- If using the MiniMind .pth format (not Qwen3), ensure you use `merge_and_unload()` before generation

**Out of memory:**
- Reduce `batch_size` from 8 to 4 or 2
- Reduce `max_seq_len` from 512 to 256
- 63.9M + batch_size=8 + bfloat16 should fit in 18GB Apple Silicon with 0.22GB used

**Slow training:**
- Apple Silicon MPS is slower than NVIDIA CUDA
- ~0.6s/step means ~40 min per epoch
- Consider using `torch.compile()` if available for speedup
