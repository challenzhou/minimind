"""
Convert user's notes (.ipynb / .md) to MiniMind pretrain jsonl format.
Usage: python convert_notes_to_jsonl.py [--notes_dir /path/to/notes] [--output dataset/notes_pretrain.jsonl] [--chunk_chars 2000]
"""
import argparse
import json
import os
import glob
import re
from tqdm import tqdm

def extract_text_from_ipynb(filepath, chunk_chars=2000):
    """Extract text from Jupyter notebook, return list of text chunks."""
    try:
        with open(filepath, encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to parse {filepath}: {e}")
        return []

    parts = []
    current = ""

    # Extract notebook title/first heading if exists
    first_heading = ""
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if src.strip():
                # Use first non-empty markdown cell as context header
                first_heading = src.split("\n")[0].strip()
                if first_heading.startswith("#"):
                    break

    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            # Strip cell magics and heavy latex for cleaner text
            src = re.sub(r'^\s*%%.*$', '', src, flags=re.MULTILINE)  # cell magics
            src = re.sub(r'\$\$.*?\$\$', '', src, flags=re.DOTALL)  # display math
            src = re.sub(r'\$([^\$]+)\$', r'\1', src)  # inline math
            src = src.strip()
            if src:
                if current:
                    current += "\n\n" + src
                else:
                    current = src
        elif cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            # Keep code comments and structure, strip outputs
            src = src.strip()
            if src:
                if current:
                    current += "\n\n[Code]\n" + src
                else:
                    current = "[Code]\n" + src

    if not current:
        return []

    # Prepend title context
    if first_heading and not current.startswith(first_heading):
        current = f"Topic: {first_heading}\n\n{current}"

    # Chunk by character limit (approximate, will be re-chunked by tokenization in training)
    chunks = []
    while len(current) > chunk_chars:
        # Try to split at paragraph or double newline
        split_point = current.rfind('\n\n', 0, chunk_chars)
        if split_point < chunk_chars // 2:
            split_point = current.rfind('\n', 0, chunk_chars)
        if split_point < chunk_chars // 4:
            split_point = current.rfind(' ', 0, chunk_chars)
        if split_point < 100:
            split_point = chunk_chars
        chunks.append(current[:split_point].strip())
        current = current[split_point:].strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def extract_text_from_md(filepath):
    """Extract text from markdown file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  [WARN] Failed to read {filepath}: {e}")
        return []

    # Strip frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2]

    # Basic cleaning
    content = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)
    content = re.sub(r'\$([^\$]+)\$', r'\1', content)
    content = content.strip()

    return [content] if content else []


def convert_notes(notes_dir, output_path, chunk_chars=2000):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    ipynbs = glob.glob(os.path.join(notes_dir, '*.ipynb'))
    mds = glob.glob(os.path.join(notes_dir, '*.md'))
    all_files = ipynbs + mds

    print(f"Found {len(ipynbs)} notebooks and {len(mds)} markdown files")

    records = []
    skipped = 0

    for filepath in tqdm(all_files, desc="Converting notes"):
        basename = os.path.basename(filepath)
        if basename.startswith('.') or basename.startswith('Untitled'):
            skipped += 1
            continue

        if filepath.endswith('.ipynb'):
            chunks = extract_text_from_ipynb(filepath, chunk_chars)
        else:
            chunks = extract_text_from_md(filepath)

        for chunk in chunks:
            if len(chunk) < 50:  # Skip very short fragments
                continue
            records.append(json.dumps({"text": chunk}, ensure_ascii=False))

    print(f"Converted {len(records)} chunks from {len(all_files) - skipped} files, skipped {skipped} files")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(records))

    print(f"Saved to {output_path}")
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert notes to MiniMind pretrain jsonl")
    parser.add_argument('--notes_dir', default='/Users/challenzhou/CodeGeeXProjects/notes',
                        help='Directory containing note files')
    parser.add_argument('--output', default='../dataset/notes_pretrain.jsonl',
                        help='Output jsonl path (relative to minimind root)')
    parser.add_argument('--chunk_chars', type=int, default=2000,
                        help='Max characters per chunk')
    args = parser.parse_args()

    # Resolve output relative to minimind root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minimind_root = os.path.dirname(script_dir)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(minimind_root, output_path)

    notes_dir = args.notes_dir
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(minimind_root, notes_dir)

    n = convert_notes(notes_dir, output_path, args.chunk_chars)
    print(f"\nDone! Generated {n} training samples.")
