import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from recursive_compressor_lm import RecursiveCompressorLM
from dataset import TextDataset


def train():
    # Hyperparameters
    text_dir = "text"
    tokenizer_name = "elyza/ELYZA-japanese-Llama-2-7b-fast"
    context_length = 4096
    d_model = 1024
    num_heads = 8
    d_ff = 2048
    chunk_size = 8
    compress_size = 4
    num_layers = 8
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = TextDataset(text_dir, tokenizer_name, context_length)
    shuffle_generator = torch.Generator()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=shuffle_generator)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Device: {device}")

    # Model
    model = RecursiveCompressorLM(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        chunk_size=chunk_size,
        compress_size=compress_size,
        num_layers=num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        shuffle_generator.manual_seed(epoch)
        ema_loss = None
        ema_beta = 0.99

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss = loss.item()
            ema_loss = batch_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.4f}", ema=f"{ema_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, EMA Loss: {ema_loss:.4f}")

    torch.save(model.state_dict(), "recursive_compressor_lm.pth")
    print("Model saved to recursive_compressor_lm.pth")


if __name__ == "__main__":
    train()
