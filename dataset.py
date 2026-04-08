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


def format_conversation_turn(query, answer):
    """対話の1ターン: [QUERY]q[ANSWER]a"""
    return f"[QUERY]{query}[ANSWER]{answer}"


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


def _tokenize_unit(tokenizer, text):
    """テキストを1ユニットとしてトークナイズ。先頭BOSのみ、末尾BOSなし。
    ユニット = [BOS] + encode(text)"""
    bos = tokenizer.bos_token_id
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [bos] + tokens


def _pack_units(units, context_length, pad_token_id, bos_token_id):
    """ユニットのリストをcontext_length長のシーケンスに詰め込む。
    各ユニットは[BOS]+tokensの形。パック結果は末尾にBOSを付加。
    結果: <s>unit1<s>unit2<s>[PAD]..."""
    packed = []
    current = []

    for unit in units:
        # +1 for trailing BOS that will be appended
        if len(unit) + 1 >= context_length:
            # Long unit: truncate to context_length (no trailing BOS)
            packed.append(unit[:context_length])
            continue

        if len(current) + len(unit) + 1 <= context_length:
            # Fits in current buffer
            current.extend(unit)
        else:
            # Flush current buffer: add trailing BOS + PAD
            current.append(bos_token_id)
            pad_len = context_length - len(current)
            packed.append(current + [pad_token_id] * pad_len)
            current = list(unit)

    # Flush remaining buffer
    if current:
        current.append(bos_token_id)
        pad_len = context_length - len(current)
        if pad_len > 0:
            packed.append(current + [pad_token_id] * pad_len)
        else:
            packed.append(current[:context_length])

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


def _build_memmap_packed(cache_path, items, tokenizer, context_length, units_fn):
    """イテレータからmemmapキャッシュを構築する（短文結合あり）。
    units_fn: item -> list of unit strings (or None to skip)
    各unitは_tokenize_unitでトークナイズされユニットのリストとして蓄積。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id

    CHUNK_SIZE = 50000
    chunk_dir = cache_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    all_units = []
    chunk_files = []
    total_items = 0

    for item in items:
        texts = units_fn(item)
        if texts is None:
            continue
        for text in texts:
            all_units.append(_tokenize_unit(tokenizer, text))
        total_items += 1

        if len(all_units) >= CHUNK_SIZE:
            packed = _pack_units(all_units, context_length, pad, bos)
            chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
            np.save(chunk_path, np.array(packed, dtype=np.uint16))
            chunk_files.append(chunk_path)
            print(f"  {total_items} items processed -> {sum(len(np.load(f)) for f in chunk_files)} packed samples", flush=True)
            all_units = []

    # Flush remaining
    if all_units:
        packed = _pack_units(all_units, context_length, pad, bos)
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
        np.save(chunk_path, np.array(packed, dtype=np.uint16))
        chunk_files.append(chunk_path)

    if not chunk_files:
        with open(cache_path + ".meta.json", "w") as f:
            json.dump({"num_samples": 0, "context_length": context_length}, f)
        import shutil
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return

    # Merge chunks into a single memmap
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


def _units_doc_item(item):
    """文書アイテム → 1ユニット"""
    return [format_document(item["text"])]


def _units_sharegpt_item(item):
    """ShareGPT対話 → ターン数ぶんのユニット"""
    turns = _extract_turns_sharegpt(item["conversations"])
    if not turns:
        return None
    return [format_conversation_turn(q, a) for q, a in turns]


def _units_messages_item(item):
    """messages対話 → ターン数ぶんのユニット"""
    turns = _extract_turns_messages(item["messages"])
    if not turns:
        return None
    return [format_conversation_turn(q, a) for q, a in turns]


def _prepare_cached_dataset(name, cache_path, tokenizer, context_length, load_fn, units_fn):
    """キャッシュがあればロード、なければ構築して返す"""
    if os.path.exists(cache_path + ".meta.json"):
        print(f"  Using cache: {cache_path}")
    else:
        print(f"  Building cache: {cache_path}")
        ds = load_fn()
        _build_memmap_packed(cache_path, ds, tokenizer, context_length, units_fn)

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
            "cache_name": "wiki_ja_v2",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        {
            "name": "shi3z/ja_conv_wikipedia_llama2pro8b_30k",
            "cache_name": "shi3z_llama2pro_v2",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_llama2pro8b_30k", split="train", cache_dir=cache_dir),
            "units": _units_sharegpt_item,
        },
        {
            "name": "shi3z/ja_conv_wikipedia_orion14B_100K",
            "cache_name": "shi3z_orion14b_v2",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir=cache_dir),
            "units": _units_sharegpt_item,
        },
    ]

    datasets = []
    for src in sources:
        print(f"Loading {src['name']}...")
        cache_path = os.path.join(mmap_dir, f"{src['cache_name']}.mmap")
        ds = _prepare_cached_dataset(
            src["name"], cache_path, tokenizer, context_length,
            src["load"], src["units"],
        )
        if ds is not None:
            datasets.append(ds)
            print(f"  Samples: {len(ds)}")

    combined = ConcatDataset(datasets)
    print(f"Total samples: {len(combined)}")
    return combined, tokenizer
