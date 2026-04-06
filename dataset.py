import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer


TOKENIZER_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast"


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_document(text):
    """文章データのフォーマット: [DOC]text"""
    return f"[DOC]{text}"


def format_conversation(turns):
    """対話データのフォーマット: [QUERY]q[ANSWER]a"""
    parts = []
    for query, answer in turns:
        parts.append(f"[QUERY]{query}[ANSWER]{answer}")
    return "".join(parts)


def _extract_turns_sharegpt(conversations):
    """shi3z形式: [{from: human/gpt, value: ...}, ...]"""
    turns = []
    i = 0
    while i + 1 < len(conversations):
        if conversations[i]["from"] == "human" and conversations[i + 1]["from"] == "gpt":
            turns.append((conversations[i]["value"], conversations[i + 1]["value"]))
            i += 2
        else:
            i += 1
    return turns


def _extract_turns_messages(messages):
    """ultrachat形式: [{role: user/assistant, content: ...}, ...]"""
    turns = []
    i = 0
    while i + 1 < len(messages):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return turns


def tokenize_with_bos(tokenizer, text, context_length):
    """BOS + text をトークナイズし、context_length以内ならBOSを末尾にも付ける。
    PADトークンで埋め、labelsのPAD部分は-100にする。"""
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # BOS + tokens (+ BOS if fits)
    if len(tokens) + 2 <= context_length:
        seq = [bos] + tokens + [bos]
    else:
        seq = [bos] + tokens

    # Truncate to context_length
    seq = seq[:context_length]

    # input = seq[:-1], label = seq[1:]
    input_len = len(seq) - 1
    input_ids = seq[:-1]
    labels = seq[1:]

    # Pad
    pad_len = context_length - 1 - input_len
    if pad_len > 0:
        input_ids = input_ids + [pad] * pad_len
        labels = labels + [-100] * pad_len

    return input_ids, labels


def _tokenize_single(tokenizer, text):
    """テキストをBOS付きでトークナイズ。末尾BOSも付ける。パディングなし。"""
    bos = tokenizer.bos_token_id
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [bos] + tokens + [bos]


def _pack_sequences(token_seqs, context_length, pad_token_id):
    """複数のトークン列をcontext_length長のシーケンスに詰め込む。
    短い文書を連結してPADの無駄を減らす。"""
    packed = []
    current = []

    for seq in token_seqs:
        if len(seq) >= context_length:
            # 長い文書はそのままtruncateして1サンプルに
            packed.append(seq[:context_length])
            continue

        if len(current) + len(seq) <= context_length:
            current.extend(seq)
        else:
            # 現在のバッファをPAD埋めして確定
            pad_len = context_length - len(current)
            packed.append(current + [pad_token_id] * pad_len)
            current = list(seq)

    # 残りのバッファ
    if current:
        pad_len = context_length - len(current)
        packed.append(current + [pad_token_id] * pad_len)

    return packed


class MemmapDataset(Dataset):
    """numpy memmapファイルからサンプルを読み出すデータセット。
    各サンプルはcontext_length長のトークン列で、__getitem__で(input_ids, labels)に変換。"""

    def __init__(self, cache_path, pad_token_id):
        with open(cache_path + ".meta.json", "r") as f:
            meta = json.load(f)
        self.num_samples = meta["num_samples"]
        self.context_length = meta["context_length"]
        self.pad_token_id = pad_token_id
        self.data = np.memmap(
            cache_path, dtype=np.uint16, mode="r",
            shape=(self.num_samples, self.context_length),
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        labels[labels == self.pad_token_id] = -100
        return input_ids, labels


def _build_memmap_packed(cache_path, items, tokenizer, context_length, format_fn):
    """イテレータからmemmapキャッシュを構築する（短文結合あり）。
    format_fn: item -> formatted text string (or None to skip)"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    pad = tokenizer.pad_token_id

    # Phase 1: tokenize all items into variable-length sequences
    CHUNK_SIZE = 50000
    chunk_dir = cache_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    token_seqs = []
    chunk_files = []
    total_items = 0

    for item in items:
        text = format_fn(item)
        if text is None:
            continue
        seq = _tokenize_single(tokenizer, text)
        token_seqs.append(seq)
        total_items += 1

        if len(token_seqs) >= CHUNK_SIZE:
            # Pack and save chunk
            packed = _pack_sequences(token_seqs, context_length, pad)
            chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
            np.save(chunk_path, np.array(packed, dtype=np.uint16))
            chunk_files.append(chunk_path)
            print(f"  {total_items} items processed -> {sum(len(np.load(f)) for f in chunk_files)} packed samples", flush=True)
            token_seqs = []

    # Flush remaining
    if token_seqs:
        packed = _pack_sequences(token_seqs, context_length, pad)
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
        np.save(chunk_path, np.array(packed, dtype=np.uint16))
        chunk_files.append(chunk_path)

    if not chunk_files:
        with open(cache_path + ".meta.json", "w") as f:
            json.dump({"num_samples": 0, "context_length": context_length}, f)
        import shutil
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return

    # Phase 2: merge chunks into a single memmap
    total_samples = sum(len(np.load(f)) for f in chunk_files)
    mmap = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(total_samples, context_length))
    offset = 0
    for chunk_path in chunk_files:
        chunk = np.load(chunk_path)
        mmap[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    mmap.flush()
    del mmap

    with open(cache_path + ".meta.json", "w") as f:
        json.dump({"num_samples": total_samples, "context_length": context_length}, f)

    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"  Cache built: {total_items} items -> {total_samples} packed samples", flush=True)


# Keep old name for compatibility
_build_memmap = _build_memmap_packed


def _format_doc_item(item):
    return format_document(item["text"])


def _format_sharegpt_item(item):
    turns = _extract_turns_sharegpt(item["conversations"])
    if not turns:
        return None
    return format_conversation(turns)


def _format_messages_item(item):
    turns = _extract_turns_messages(item["messages"])
    if not turns:
        return None
    return format_conversation(turns)


def _prepare_cached_dataset(name, cache_path, tokenizer, context_length, load_fn, format_fn):
    """キャッシュがあればロード、なければ構築して返す"""
    if os.path.exists(cache_path + ".meta.json"):
        print(f"  Using cache: {cache_path}")
    else:
        print(f"  Building cache: {cache_path}")
        ds = load_fn()
        _build_memmap_packed(cache_path, ds, tokenizer, context_length, format_fn)

    with open(cache_path + ".meta.json", "r") as f:
        meta = json.load(f)
    if meta["num_samples"] == 0:
        return None

    return MemmapDataset(cache_path, tokenizer.pad_token_id)


def prepare_all_datasets(context_length, cache_dir=None):
    """全データセットを準備して結合する（日本語のみ）"""
    tokenizer = get_tokenizer()
    if cache_dir is None:
        cache_dir = "./data/hf_cache"
    mmap_dir = os.path.join(cache_dir, "mmap")

    sources = [
        {
            "name": "wikimedia/wikipedia (ja)",
            "cache_name": "wiki_ja_packed",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir=cache_dir),
            "format": _format_doc_item,
        },
        {
            "name": "shi3z/ja_conv_wikipedia_llama2pro8b_30k",
            "cache_name": "shi3z_llama2pro_packed",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_llama2pro8b_30k", split="train", cache_dir=cache_dir),
            "format": _format_sharegpt_item,
        },
        {
            "name": "shi3z/ja_conv_wikipedia_orion14B_100K",
            "cache_name": "shi3z_orion14b_packed",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir=cache_dir),
            "format": _format_sharegpt_item,
        },
    ]

    datasets = []
    for src in sources:
        print(f"Loading {src['name']}...")
        cache_path = os.path.join(mmap_dir, f"{src['cache_name']}.mmap")
        ds = _prepare_cached_dataset(
            src["name"], cache_path, tokenizer, context_length,
            src["load"], src["format"],
        )
        if ds is not None:
            datasets.append(ds)
            print(f"  Samples: {len(ds)}")

    combined = ConcatDataset(datasets)
    print(f"Total samples: {len(combined)}")
    return combined, tokenizer
