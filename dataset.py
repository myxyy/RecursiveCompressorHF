import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, concatenate_datasets
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


class TokenizedDataset(Dataset):
    """事前トークナイズ済みデータセット。各サンプルは (input_ids, labels) のペア。"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, labels = self.data[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def prepare_document_dataset(dataset_name, subset, split, tokenizer, context_length, cache_dir=None):
    """文章データセットを読み込みトークナイズする"""
    if subset:
        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    data = []
    for item in ds:
        text = format_document(item["text"])
        input_ids, labels = tokenize_with_bos(tokenizer, text, context_length)
        data.append((input_ids, labels))

    return TokenizedDataset(data)


def prepare_conversation_dataset_sharegpt(dataset_name, split, tokenizer, context_length, cache_dir=None):
    """ShareGPT形式の対話データセットを読み込みトークナイズする"""
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    data = []
    for item in ds:
        turns = _extract_turns_sharegpt(item["conversations"])
        if not turns:
            continue
        text = format_conversation(turns)
        input_ids, labels = tokenize_with_bos(tokenizer, text, context_length)
        data.append((input_ids, labels))

    return TokenizedDataset(data)


def prepare_conversation_dataset_messages(dataset_name, split, tokenizer, context_length, cache_dir=None):
    """messages形式の対話データセットを読み込みトークナイズする"""
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    data = []
    for item in ds:
        turns = _extract_turns_messages(item["messages"])
        if not turns:
            continue
        text = format_conversation(turns)
        input_ids, labels = tokenize_with_bos(tokenizer, text, context_length)
        data.append((input_ids, labels))

    return TokenizedDataset(data)


def prepare_all_datasets(context_length, cache_dir=None):
    """全データセットを準備して結合する"""
    tokenizer = get_tokenizer()

    datasets = []

    # Document datasets
    print("Loading wikimedia/wikipedia (ja)...")
    datasets.append(prepare_document_dataset(
        "wikimedia/wikipedia", "20231101.ja", "train", tokenizer, context_length, cache_dir))

    print("Loading wikimedia/wikipedia (en)...")
    datasets.append(prepare_document_dataset(
        "wikimedia/wikipedia", "20231101.en", "train", tokenizer, context_length, cache_dir))

    print("Loading JeanKaddour/minipile...")
    datasets.append(prepare_document_dataset(
        "JeanKaddour/minipile", None, "train", tokenizer, context_length, cache_dir))

    # Conversation datasets (ShareGPT format)
    print("Loading shi3z/ja_conv_wikipedia_llama2pro8b_30k...")
    datasets.append(prepare_conversation_dataset_sharegpt(
        "shi3z/ja_conv_wikipedia_llama2pro8b_30k", "train", tokenizer, context_length, cache_dir))

    print("Loading shi3z/ja_conv_wikipedia_orion14B_100K...")
    datasets.append(prepare_conversation_dataset_sharegpt(
        "shi3z/ja_conv_wikipedia_orion14B_100K", "train", tokenizer, context_length, cache_dir))

    # Conversation datasets (messages format)
    print("Loading HuggingFaceH4/ultrachat_200k...")
    datasets.append(prepare_conversation_dataset_messages(
        "HuggingFaceH4/ultrachat_200k", "train_sft", tokenizer, context_length, cache_dir))

    combined = ConcatDataset(datasets)
    print(f"Total samples: {len(combined)}")
    return combined, tokenizer
