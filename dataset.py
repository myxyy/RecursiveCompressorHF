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


def _text_to_chunks(tokenizer, text, context_length):
    """テキストをBOS付きトークン列にして、context_length単位で分割する。
    最初のチャンクは[BOS]+tokens、続きはtokens（継続チャンク、BOSなし）。
    各チャンクはcontext_length以下の長さ。"""
    bos = tokenizer.bos_token_id
    tokens = [bos] + tokenizer.encode(text, add_special_tokens=False)
    return [tokens[i:i + context_length] for i in range(0, len(tokens), context_length)]


def _pack_chunks(chunks, context_length, pad_token_id):
    """チャンクをcontext_length長のサンプルに詰め込む。
    各チャンクはそのまま連結（末尾BOSは追加しない；次チャンクの先頭BOSが区切り役）。
    不足分はPADで埋める。返すリストの全要素は必ずcontext_length長。"""
    packed = []
    current = []

    def _flush():
        seq = (current + [pad_token_id] * context_length)[:context_length]
        packed.append(seq)

    for chunk in chunks:
        assert len(chunk) <= context_length, \
            f"Chunk exceeds context_length: {len(chunk)} > {context_length}"
        if len(current) + len(chunk) > context_length:
            _flush()
            current = list(chunk)
        else:
            current.extend(chunk)

    if current:
        _flush()

    assert all(len(s) == context_length for s in packed), \
        f"Pack length mismatch: {set(len(s) for s in packed)}, expected {context_length}"
    return packed


class MemmapDataset(Dataset):
    """numpy memmapファイルからサンプルを読み出すデータセット。
    各サンプルはcontext_length長のトークン列で、__getitem__で(input_ids, labels)に変換。
    prefault=Trueでファイル全体を事前にOSページキャッシュへ読み込む
    （ディスクI/O削減。複数プロセス間でページキャッシュは共有されるため安全）。"""

    def __init__(self, cache_path, pad_token_id, prefault=False):
        with open(cache_path + ".meta.json", "r") as f:
            meta = json.load(f)
        self.num_samples = meta["num_samples"]
        self.context_length = meta["context_length"]
        self.pad_token_id = pad_token_id
        self.data = np.memmap(
            cache_path, dtype=np.uint16, mode="r",
            shape=(self.num_samples, self.context_length),
        )
        if prefault:
            # Touch all pages to populate the OS page cache.
            # Page cache is shared across processes, so total RAM use stays
            # bounded regardless of how many ranks call this.
            _ = int(self.data.sum())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        labels[labels == self.pad_token_id] = -100
        return input_ids, labels


def _build_memmap_packed(cache_path, items, tokenizer, context_length, units_fn):
    """イテレータからmemmapキャッシュを構築する。
    units_fn: item -> list of text strings (or None to skip)
    各テキストはBOS付きトークン列にしてcontext_length単位で分割し、パックする。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    pad = tokenizer.pad_token_id

    CHUNK_SIZE = 50000
    chunk_dir = cache_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    pending = []
    chunk_files = []
    total_items = 0

    for item in items:
        texts = units_fn(item)
        if texts is None:
            continue
        for text in texts:
            pending.extend(_text_to_chunks(tokenizer, text, context_length))
        total_items += 1

        if len(pending) >= CHUNK_SIZE:
            packed = _pack_chunks(pending, context_length, pad)
            chunk_arr = np.stack([np.array(s, dtype=np.uint16) for s in packed])
            chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
            np.save(chunk_path, chunk_arr)
            chunk_files.append(chunk_path)
            print(f"  {total_items} items processed -> {sum(len(np.load(f)) for f in chunk_files)} packed samples", flush=True)
            pending = []

    # Flush remaining
    if pending:
        packed = _pack_chunks(pending, context_length, pad)
        chunk_arr = np.stack([np.array(s, dtype=np.uint16) for s in packed])
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
        np.save(chunk_path, chunk_arr)
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


def _units_doc_item(item):
    """文書アイテム → 1テキスト（[DOC]プリフィックスなし、生のtext）"""
    return [item["text"]]


def _units_sharegpt_item(item):
    """ShareGPT対話 → ターン数ぶんのユニット文字列"""
    turns = _extract_turns_sharegpt(item["conversations"])
    if not turns:
        return None
    return [format_conversation_turn(q, a) for q, a in turns]


def _units_messages_item(item):
    """messages対話 → ターン数ぶんのユニット文字列"""
    turns = _extract_turns_messages(item["messages"])
    if not turns:
        return None
    return [format_conversation_turn(q, a) for q, a in turns]


def _prepare_cached_dataset(name, cache_path, tokenizer, context_length, load_fn, units_fn, prefault=False):
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

    return MemmapDataset(cache_path, tokenizer.pad_token_id, prefault=prefault)


DATASET_TYPES = ("pretrain", "instruct")


def _all_sources(cache_dir):
    """全ソース定義。dataset_typeで絞り込んで使う。"""
    return {
        "wiki_ja": {
            "name": "wikimedia/wikipedia (ja)",
            "cache_name": "wiki_ja_v3",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "wiki_en": {
            "name": "wikimedia/wikipedia (en)",
            "cache_name": "wiki_en_v3",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.en", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "cc100_ja": {
            "name": "hotchpotch/cc100-ja-documents",
            "cache_name": "cc100_ja_v3",
            "load": lambda: load_dataset("hotchpotch/cc100-ja-documents", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "minipile": {
            "name": "JeanKaddour/minipile",
            "cache_name": "minipile_v3",
            "load": lambda: load_dataset("JeanKaddour/minipile", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "shi3z_llama2pro": {
            "name": "shi3z/ja_conv_wikipedia_llama2pro8b_30k",
            "cache_name": "shi3z_llama2pro_v3",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_llama2pro8b_30k", split="train", cache_dir=cache_dir),
            "units": _units_sharegpt_item,
        },
        "shi3z_orion14b": {
            "name": "shi3z/ja_conv_wikipedia_orion14B_100K",
            "cache_name": "shi3z_orion14b_v3",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir=cache_dir),
            "units": _units_sharegpt_item,
        },
        "ultrachat": {
            "name": "HuggingFaceH4/ultrachat_200k",
            "cache_name": "ultrachat_v3",
            "load": lambda: load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=cache_dir),
            "units": _units_messages_item,
        },
    }


_DATASET_GROUPS = {
    "pretrain": ["wiki_ja", "wiki_en", "cc100_ja", "minipile"],
    "instruct": ["shi3z_llama2pro", "shi3z_orion14b", "ultrachat"],
}


def prepare_all_datasets(context_length, cache_dir=None, prefault=False, dataset_type="pretrain"):
    """データセット種別に応じて構成データセットを準備し結合する。
    dataset_type:
        "pretrain" - wiki_ja + wiki_en + cc100_ja + minipile（文書のみ）
        "instruct" - shi3z 2種 + ultrachat_200k（対話のみ）
    prefault=Trueでキャッシュ全体をOSページキャッシュに事前読み込み
    （ディスクI/O削減。プロセス間でページキャッシュ共有のため安全）。"""
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}")

    tokenizer = get_tokenizer()
    if cache_dir is None:
        cache_dir = "./data/hf_cache"
    mmap_dir = os.path.join(cache_dir, "mmap")

    all_sources = _all_sources(cache_dir)
    source_keys = _DATASET_GROUPS[dataset_type]

    print(f"Dataset type: {dataset_type}")
    datasets = []
    for key in source_keys:
        src = all_sources[key]
        print(f"Loading {src['name']}...")
        cache_path = os.path.join(mmap_dir, f"{src['cache_name']}.mmap")
        ds = _prepare_cached_dataset(
            src["name"], cache_path, tokenizer, context_length,
            src["load"], src["units"], prefault=prefault,
        )
        if ds is not None:
            datasets.append(ds)
            print(f"  Samples: {len(ds)}{' (prefaulted)' if prefault else ''}")

    combined = ConcatDataset(datasets)
    print(f"Total samples: {len(combined)}")
    return combined, tokenizer
