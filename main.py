from recursive_compressor import RecursiveCompressor
import torch

def main():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    chunk_size = 4
    compress_size = 2
    seq_len = 1024
    batch_size = 2

    model = RecursiveCompressor(d_model, num_heads, d_ff, chunk_size, compress_size)
    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)
    print(output.shape)

    hidden = None
    x = torch.randn(batch_size, d_model)
    for _ in range(1000):
        x, hidden = model.predict(x, hidden)
        print(len(hidden), x.mean().item())

if __name__ == "__main__":
    main()
