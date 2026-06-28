import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
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
    """テキストを `<s>...</s>` 形式のトークン列にして、context_length単位で分割する。
    最初のチャンクは [BOS]+tokens で始まり、最後のチャンクは末尾に [EOS] が付く
    (1チャンクに収まる短いテキストは [BOS]+tokens+[EOS] になる)。継続チャンクは
    マーカーなし。各チャンクは context_length 以下の長さ。"""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    tokens = [bos] + tokenizer.encode(text, add_special_tokens=False) + [eos]
    return [tokens[i:i + context_length] for i in range(0, len(tokens), context_length)]


def _pack_chunks(chunks, context_length, pad_token_id):
    """チャンクをcontext_length長のサンプルに詰め込む。
    各チャンクはそのまま連結（末尾BOSは追加しない；次チャンクの先頭BOSが区切り役）。
    不足分はPADで埋める。返すリストの全要素は必ずcontext_length長。"""
    packed = []
    current = []

    # Need at least 2 content tokens so that labels (= seq[1:]) has 1+ valid
    # (non-PAD) position. A 1-content-token sample yields all-PAD labels which
    # makes CrossEntropyLoss return NaN (mean of empty set).
    MIN_CONTENT = 2

    def _flush():
        if len(current) < MIN_CONTENT:
            return  # Drop too-short samples to avoid all-PAD-label NaN downstream
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


def _conversation_to_ids_and_mask(tokenizer, turns):
    """対話を Llama2 形式トークン列と応答 loss マスクに変換する。

    各ターン (q, a) を `<s>[INST]q[/INST]` (prompt) と `a</s>` (answer) に分けて
    トークナイズし、連結すると `<s>[INST]q1[/INST]a1</s><s>[INST]q2[/INST]a2</s>...`
    になる。マスクは応答トークン (answer 本文 + その EOS) のみ 1、prompt 部
    (BOS + [INST]q[/INST]) は 0。prompt と answer を別々にトークナイズするのは
    マスク境界を正確にし、推論時 (chat_server: `<s>[INST]q[/INST]` を入力して
    応答を生成) と一致させるため。

    Returns: (ids, mask) — どちらも同じ長さの Python list。"""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    ids, mask = [], []
    for q, a in turns:
        prompt_ids = [bos] + tokenizer.encode(f"[INST]{q}[/INST]", add_special_tokens=False)
        answer_ids = tokenizer.encode(a, add_special_tokens=False) + [eos]
        ids.extend(prompt_ids)
        mask.extend([0] * len(prompt_ids))
        ids.extend(answer_ids)
        mask.extend([1] * len(answer_ids))
    return ids, mask


def _conversation_to_samples(tokenizer, turns, context_length):
    """1会話を context_length 長サンプルへ分割する（会話をまたぐ連結はしない）。
    各サンプルは (ids, mask) で長さ context_length（不足は PAD / mask=0 で埋める）。
    ラベル位置 (mask[1:]) に応答トークンが1つも無いチャンクは捨てる（全 -100 で
    CE loss が 0/0=NaN になるのを防ぐ）。"""
    pad = tokenizer.pad_token_id
    ids, mask = _conversation_to_ids_and_mask(tokenizer, turns)

    samples = []
    for i in range(0, len(ids), context_length):
        chunk_ids = ids[i:i + context_length]
        chunk_mask = mask[i:i + context_length]
        if sum(chunk_mask[1:]) == 0:
            continue  # no answer token in a label position
        pad_len = context_length - len(chunk_ids)
        chunk_ids = chunk_ids + [pad] * pad_len
        chunk_mask = chunk_mask + [0] * pad_len
        samples.append((chunk_ids, chunk_mask))
    return samples


class MemmapDataset(Dataset):
    """numpy memmapファイルからサンプルを読み出すデータセット。
    各サンプルはcontext_length長のトークン列で、__getitem__で(input_ids, labels)に変換。
    prefault=Trueでファイル全体を事前にOSページキャッシュへ読み込む
    （ディスクI/O削減。複数プロセス間でページキャッシュは共有されるため安全）。

    `has_mask` が立っているキャッシュ（instruct）では並列の `.mask` memmap を読み、
    応答トークン (mask=1) の位置だけ loss を残す（それ以外は -100）。マスクが無い
    キャッシュ（pretrain）では従来通り PAD 以外の全位置で loss を取る。"""

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
        self.mask = None
        mask_path = cache_path + ".mask"
        if meta.get("has_mask") and os.path.exists(mask_path):
            self.mask = np.memmap(
                mask_path, dtype=np.uint8, mode="r",
                shape=(self.num_samples, self.context_length),
            )
        if prefault:
            # Touch all pages to populate the OS page cache.
            # Page cache is shared across processes, so total RAM use stays
            # bounded regardless of how many ranks call this.
            _ = int(self.data.sum())
            if self.mask is not None:
                _ = int(self.mask.sum())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        if self.mask is not None:
            # Keep loss only where the *target* token (seq[1:]) is an answer
            # token. PAD positions have mask=0, so they are masked too.
            m = torch.from_numpy(self.mask[idx].astype(np.int64))
            labels[m[1:] == 0] = -100
        else:
            labels[labels == self.pad_token_id] = -100
        return input_ids, labels


# Worker-process state for parallel tokenization. Initialized via
# ProcessPoolExecutor's `initializer` so the tokenizer is loaded only once
# per worker process.
_WORKER_TOKENIZER = None


def _tokenizer_init_worker(tokenizer_name):
    global _WORKER_TOKENIZER
    tk = AutoTokenizer.from_pretrained(tokenizer_name)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    _WORKER_TOKENIZER = tk


def _tokenize_text_lists_worker(text_lists, context_length):
    """Worker: tokenize a batch of text lists using the worker-local tokenizer.
    text_lists: list of (list[str] or None)
    Returns: flat list of chunk token-id lists."""
    chunks = []
    for texts in text_lists:
        if texts is None:
            continue
        for text in texts:
            chunks.extend(_text_to_chunks(_WORKER_TOKENIZER, text, context_length))
    return chunks


def _tokenize_conversations_worker(turns_lists, context_length):
    """Worker: convert a batch of conversations (each a list of (q, a) tuples or
    None) into padded (ids, mask) samples using the worker-local tokenizer.
    Returns: flat list of (ids_list, mask_list) tuples."""
    samples = []
    for turns in turns_lists:
        if not turns:
            continue
        samples.extend(_conversation_to_samples(_WORKER_TOKENIZER, turns, context_length))
    return samples


def _items_to_text_batches(items, units_fn, batch_size):
    """Generator yielding lists of (units_fn(item) for item in batch)."""
    batch = []
    for item in items:
        batch.append(units_fn(item))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_memmap_packed(cache_path, items, tokenizer, context_length, units_fn, num_workers=1):
    """イテレータからmemmapキャッシュを構築する。
    units_fn: item -> list of text strings (or None to skip)
    各テキストは `<s>...</s>` 形式 (BOS+tokens+EOS) のトークン列にして
    context_length単位で分割し、パックする。
    num_workers > 1 で並列トークナイズを使用（順序は保たれる）。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    pad = tokenizer.pad_token_id

    PACK_THRESHOLD = 50000  # accumulated chunks before flushing to npy
    BATCH_ITEMS = 500       # items per worker call (only used in parallel mode)
    chunk_dir = cache_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    pending = []
    chunk_files = []
    total_items = 0

    def _flush_pack():
        nonlocal pending
        if not pending:
            return
        packed = _pack_chunks(pending, context_length, pad)
        chunk_arr = np.stack([np.array(s, dtype=np.uint16) for s in packed])
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_files)}.npy")
        np.save(chunk_path, chunk_arr)
        chunk_files.append(chunk_path)
        pending = []

    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_tokenizer_init_worker,
            initargs=(TOKENIZER_NAME,),
        ) as executor:
            in_flight = []  # list of (future, batch_size)
            max_in_flight = num_workers * 2
            batch_iter = _items_to_text_batches(items, units_fn, BATCH_ITEMS)
            iter_done = False

            while not iter_done or in_flight:
                # Refill the in-flight queue
                while not iter_done and len(in_flight) < max_in_flight:
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        iter_done = True
                        break
                    fut = executor.submit(_tokenize_text_lists_worker, batch, context_length)
                    in_flight.append((fut, len(batch)))

                if in_flight:
                    fut, bs = in_flight.pop(0)
                    pending.extend(fut.result())
                    total_items += bs

                    if len(pending) >= PACK_THRESHOLD:
                        _flush_pack()
                        print(f"  {total_items} items processed -> {sum(len(np.load(f)) for f in chunk_files)} packed samples", flush=True)
    else:
        for item in items:
            texts = units_fn(item)
            if texts is None:
                continue
            for text in texts:
                pending.extend(_text_to_chunks(tokenizer, text, context_length))
            total_items += 1

            if len(pending) >= PACK_THRESHOLD:
                _flush_pack()
                print(f"  {total_items} items processed -> {sum(len(np.load(f)) for f in chunk_files)} packed samples", flush=True)

    # Final flush
    _flush_pack()

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


def _build_memmap_conversations(cache_path, items, tokenizer, context_length, turns_fn, num_workers=1):
    """対話用 memmap キャッシュを構築する（instruct）。

    pretrain 用の `_build_memmap_packed` と違い、**会話をまたぐ短文結合をしない**:
    1会話を（必要なら複数の）context_length 長サンプルへ分割し、各サンプルは独立。
    トークンIDを `cache_path` (.mmap, uint16) に、応答 loss マスクを
    `cache_path + ".mask"` (uint8) に保存し、meta に `has_mask=True` を記録する。

    turns_fn: item -> list of (query, answer) tuples (or None to skip)
    num_workers > 1 で並列トークナイズ（順序は保たれる）。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mask_path = cache_path + ".mask"

    FLUSH_THRESHOLD = 50000  # accumulated samples before flushing to npy
    BATCH_ITEMS = 500        # conversations per worker call (parallel mode only)
    chunk_dir = cache_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    pending = []      # list of (ids_list, mask_list)
    chunk_files = []  # list of (ids_npy_path, mask_npy_path)
    total_items = 0

    def _flush():
        nonlocal pending
        if not pending:
            return
        ids_arr = np.stack([np.array(s[0], dtype=np.uint16) for s in pending])
        mask_arr = np.stack([np.array(s[1], dtype=np.uint8) for s in pending])
        ids_p = os.path.join(chunk_dir, f"ids_{len(chunk_files)}.npy")
        mask_p = os.path.join(chunk_dir, f"mask_{len(chunk_files)}.npy")
        np.save(ids_p, ids_arr)
        np.save(mask_p, mask_arr)
        chunk_files.append((ids_p, mask_p))
        pending = []

    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_tokenizer_init_worker,
            initargs=(TOKENIZER_NAME,),
        ) as executor:
            in_flight = []
            max_in_flight = num_workers * 2
            batch_iter = _items_to_text_batches(items, turns_fn, BATCH_ITEMS)
            iter_done = False

            while not iter_done or in_flight:
                while not iter_done and len(in_flight) < max_in_flight:
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        iter_done = True
                        break
                    fut = executor.submit(_tokenize_conversations_worker, batch, context_length)
                    in_flight.append((fut, len(batch)))

                if in_flight:
                    fut, bs = in_flight.pop(0)
                    pending.extend(fut.result())
                    total_items += bs

                    if len(pending) >= FLUSH_THRESHOLD:
                        _flush()
                        print(f"  {total_items} conversations processed -> {sum(len(np.load(f[0])) for f in chunk_files)} samples", flush=True)
    else:
        for item in items:
            turns = turns_fn(item)
            if turns:
                pending.extend(_conversation_to_samples(tokenizer, turns, context_length))
            total_items += 1

            if len(pending) >= FLUSH_THRESHOLD:
                _flush()
                print(f"  {total_items} conversations processed -> {sum(len(np.load(f[0])) for f in chunk_files)} samples", flush=True)

    _flush()

    if not chunk_files:
        with open(cache_path + ".meta.json", "w") as f:
            json.dump({"num_samples": 0, "context_length": context_length, "has_mask": True}, f)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return

    total_samples = sum(len(np.load(f[0])) for f in chunk_files)
    ids_mmap = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(total_samples, context_length))
    mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode="w+", shape=(total_samples, context_length))
    offset = 0
    for ids_p, mask_p in chunk_files:
        ic = np.load(ids_p)
        mc = np.load(mask_p)
        ids_mmap[offset:offset + len(ic)] = ic
        mask_mmap[offset:offset + len(mc)] = mc
        offset += len(ic)
    ids_mmap.flush()
    mask_mmap.flush()
    del ids_mmap, mask_mmap

    with open(cache_path + ".meta.json", "w") as f:
        json.dump({"num_samples": total_samples, "context_length": context_length, "has_mask": True}, f)

    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"  Cache built: {total_items} conversations -> {total_samples} samples", flush=True)


def _units_doc_item(item):
    """文書アイテム → 1テキスト（[DOC]プリフィックスなし、生のtext）"""
    return [item["text"]]


def _turns_sharegpt_item(item):
    """ShareGPT対話 → (query, answer) タプルのリスト（空ならNone）"""
    turns = _extract_turns_sharegpt(item["conversations"])
    return turns if turns else None


def _turns_messages_item(item):
    """messages対話 → (query, answer) タプルのリスト（空ならNone）"""
    turns = _extract_turns_messages(item["messages"])
    return turns if turns else None


def _prepare_cached_dataset(name, cache_path, tokenizer, context_length, load_fn, units_fn,
                            prefault=False, num_workers=1, conversational=False):
    """キャッシュがあればロード、なければ構築して返す。
    conversational=True で対話用ビルダー（短文結合なし・応答マスク付き）を使う。"""
    if os.path.exists(cache_path + ".meta.json"):
        print(f"  Using cache: {cache_path}")
    else:
        print(f"  Building cache: {cache_path} (num_workers={num_workers})")
        ds = load_fn()
        if conversational:
            _build_memmap_conversations(cache_path, ds, tokenizer, context_length, units_fn, num_workers=num_workers)
        else:
            _build_memmap_packed(cache_path, ds, tokenizer, context_length, units_fn, num_workers=num_workers)

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
            "cache_name": "wiki_ja_v5",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "wiki_en": {
            "name": "wikimedia/wikipedia (en)",
            "cache_name": "wiki_en_v5",
            "load": lambda: load_dataset("wikimedia/wikipedia", "20231101.en", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "cc100_ja": {
            "name": "hotchpotch/cc100-ja-documents",
            "cache_name": "cc100_ja_v5",
            "load": lambda: load_dataset("hotchpotch/cc100-ja-documents", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "minipile": {
            "name": "JeanKaddour/minipile",
            "cache_name": "minipile_v5",
            "load": lambda: load_dataset("JeanKaddour/minipile", split="train", cache_dir=cache_dir),
            "units": _units_doc_item,
        },
        "shi3z_llama2pro": {
            "name": "shi3z/ja_conv_wikipedia_llama2pro8b_30k",
            "cache_name": "shi3z_llama2pro_v6",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_llama2pro8b_30k", split="train", cache_dir=cache_dir),
            "units": _turns_sharegpt_item,
        },
        "shi3z_orion14b": {
            "name": "shi3z/ja_conv_wikipedia_orion14B_100K",
            "cache_name": "shi3z_orion14b_v6",
            "load": lambda: load_dataset("shi3z/ja_conv_wikipedia_orion14B_100K", split="train", cache_dir=cache_dir),
            "units": _turns_sharegpt_item,
        },
        "ultrachat": {
            "name": "HuggingFaceH4/ultrachat_200k",
            "cache_name": "ultrachat_v6",
            "load": lambda: load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=cache_dir),
            "units": _turns_messages_item,
        },
    }


_DATASET_GROUPS = {
    "pretrain": ["wiki_ja", "wiki_en", "cc100_ja", "minipile"],
    "instruct": ["shi3z_llama2pro", "shi3z_orion14b", "ultrachat"],
}


def prepare_all_datasets(context_length, cache_dir=None, prefault=False, dataset_type="pretrain",
                         num_workers=1):
    """データセット種別に応じて構成データセットを準備し結合する。
    dataset_type:
        "pretrain" - wiki_ja + wiki_en + cc100_ja + minipile（文書のみ）
        "instruct" - shi3z 2種 + ultrachat_200k（対話のみ）
    prefault=Trueでキャッシュ全体をOSページキャッシュに事前読み込み
    （ディスクI/O削減。プロセス間でページキャッシュ共有のため安全）。
    num_workers > 1 でキャッシュ未構築のデータセットを並列トークナイズ。"""
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}")

    tokenizer = get_tokenizer()
    if cache_dir is None:
        cache_dir = "./data/hf_cache"
    mmap_dir = os.path.join(cache_dir, "mmap")

    all_sources = _all_sources(cache_dir)
    source_keys = _DATASET_GROUPS[dataset_type]

    # instruct: 会話単位サンプル化（短文結合なし）＋応答のみ loss マスク。
    # pretrain: 従来通り短文結合してパック、PAD以外の全位置で loss。
    conversational = (dataset_type == "instruct")

    print(f"Dataset type: {dataset_type}")
    datasets = []
    for key in source_keys:
        src = all_sources[key]
        print(f"Loading {src['name']}...")
        cache_path = os.path.join(mmap_dir, f"{src['cache_name']}.mmap")
        ds = _prepare_cached_dataset(
            src["name"], cache_path, tokenizer, context_length,
            src["load"], src["units"], prefault=prefault, num_workers=num_workers,
            conversational=conversational,
        )
        if ds is not None:
            datasets.append(ds)
            print(f"  Samples: {len(ds)}{' (prefaulted)' if prefault else ''}")

    combined = ConcatDataset(datasets)
    print(f"Total samples: {len(combined)}")
    return combined, tokenizer
