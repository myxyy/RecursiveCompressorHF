import glob
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, text_dir, tokenizer_name, context_length):
        self.context_length = context_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        texts = []
        for path in sorted(glob.glob(f"{text_dir}/*.txt")):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        full_text = "\n".join(texts)

        self.token_ids = self.tokenizer.encode(full_text)
        num_tokens = len(self.token_ids)
        # +1 because we need input and target shifted by 1
        self.num_samples = (num_tokens - 1) // context_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1
        chunk = self.token_ids[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
