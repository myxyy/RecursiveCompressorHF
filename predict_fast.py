import argparse
import torch
from transformers import AutoTokenizer
from recursive_compressor_lm import RecursiveCompressorLM


def predict(prompt, context_length=1024, temperature=1.0, weights_path="recursive_compressor_lm.pth"):
    tokenizer_name = "elyza/ELYZA-japanese-Llama-2-7b-fast"
    d_model = 1024
    num_heads = 8
    d_ff = 2048
    chunk_size = 8
    compress_size = 4
    num_layers = 8

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
    generated = list(token_ids)

    with torch.no_grad():
        hidden = None
        # Feed prompt tokens one by one to build hidden state
        for token_id in token_ids:
            input_id = torch.tensor([token_id], dtype=torch.long, device=device)
            logits, hidden = model.predict(input_id, hidden)

        # Generate new tokens
        while len(generated) < context_length:
            next_logits = logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

            logits, hidden = model.predict(next_token, hidden)

    generated_text = tokenizer.decode(generated)
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
