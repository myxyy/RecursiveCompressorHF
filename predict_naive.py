import argparse
import torch
from transformers import AutoTokenizer
from recursive_compressor_lm import RecursiveCompressorLM


def predict(prompt, context_length=1024, temperature=1.0, weights_path="recursive_compressor_lm.pth"):
    tokenizer_name = "elyza/ELYZA-japanese-Llama-2-7b-fast"
    d_model = 512
    num_heads = 8
    d_ff = 2048
    chunk_size = 8
    compress_size = 4
    num_layers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = RecursiveCompressorLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        chunk_size=chunk_size,
        compress_size=compress_size,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        while input_ids.size(1) < context_length:
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="入力テキスト")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--weights", type=str, default="recursive_compressor_lm.pth", help="重みファイルのパス")
    args = parser.parse_args()

    text = predict(args.prompt, args.context_length, args.temperature, args.weights)
    print(text)


if __name__ == "__main__":
    main()
