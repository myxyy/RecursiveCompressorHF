import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from recursive_compressor_lm import RecursiveCompressorLM
from dataset import TextDataset


def train():
    # Hyperparameters
    text_dir = "text"
    tokenizer_name = "elyza/ELYZA-japanese-Llama-2-7b-fast"
    context_length = 2048
    d_model = 512
    num_heads = 8
    d_ff = 2048
    chunk_size = 8
    compress_size = 4
    num_layers = 4
    batch_size = 4
    num_epochs = 10
    learning_rate = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = TextDataset(text_dir, tokenizer_name, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "recursive_compressor_lm.pth")
    print("Model saved to recursive_compressor_lm.pth")


if __name__ == "__main__":
    train()
