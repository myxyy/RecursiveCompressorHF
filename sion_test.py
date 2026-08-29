import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =========================================================================
# 1. LLM Token Embedding Wrapper
# =========================================================================
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, backbone_model):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.backbone = backbone_model
        self.rms_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)
        out = self.backbone(x)
        if out.dim() == 4:
            out = out.mean(dim=0)
        out = self.rms_norm(out)
        logits = self.lm_head(out)
        return logits


# =========================================================================
# 2. Synthetic Needle-in-a-Haystack 
# =========================================================================
def generate_needle_batch(batch_size, context_length, vocab_size, device):
    """
    길이가 context_length인 무작위 토큰 시퀀스를 만들고,
    임의의 위치에 Key-Value 바늘을 주입한 뒤, 맨 끝에 Query 토큰을 채워 넣습니다.
    """
    needle_key = 7777
    needle_val = 8888
    query_token = 9999

    tokens = torch.randint(low=10, high=vocab_size - 100, size=(batch_size, context_length), device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(batch_size):
        needle_pos = torch.randint(low=int(context_length * 0.1), high=int(context_length * 0.8), size=(1,)).item()

        val_id = torch.randint(low=1000, high=5000, size=(1,)).item()
        
        tokens[i, needle_pos] = needle_key
        tokens[i, needle_pos + 1] = val_id
        tokens[i, -1] = query_token 
        
        targets[i] = val_id 
    return tokens, targets



def train_needle_retrieval(model, vocab_size, device, steps=500, batch_size=16, h_dim=512):

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_phases = [32, 512]
    
    print("\n" + "="*60)
    print("      STARTING ARIS-NET LONG-CONTEXT NEEDLE TRAINING")
    print("="*60)

    model.train()
    
    for phase_len in train_phases:
        print(f"\n[Training Phase] Curriculum Context Length: {phase_len} Tokens")
        pbar = tqdm(range(steps), desc=f"Phase {phase_len}")
        
        for step in pbar:
            inputs, targets = generate_needle_batch(batch_size, phase_len, vocab_size, device)
            
            optimizer.zero_grad()
            logits = model(inputs) 

            pred_logits = logits[:, -1, :] 
            loss = criterion(pred_logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item() * 100
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Train Acc": f"{acc:.1f}%"})

            if acc > 98.0 and step > 100:
                print(f" Phase {phase_len} Early Convergence Reached at Step {step}!")
                break

    print("\n✔ Training Completed Successfully!")


# =========================================================================
# 4. Long-Context Needle Retrieval evaluate
# =========================================================================
def evaluate_needle_in_a_haystack(model, vocab_size, context_lengths, device, num_trials=20):
    model.eval()
    print("\n" + "="*60)
    print("      EVALUATING NEEDLE IN A HAYSTACK (Out-of-Distribution Test)")
    print("="*60)

    results = {}
    with torch.no_grad():
        for length in context_lengths:
            inputs, targets = generate_needle_batch(num_trials, length, vocab_size, device)
            logits = model(inputs)
            
            pred_tokens = logits[:, -1, :].argmax(dim=-1)
            correct = (pred_tokens == targets).sum().item()
            
            accuracy = (correct / num_trials) * 100
            results[length] = accuracy
            print(f"Context Length: {length:6d} tokens | Retrieval Acc: {accuracy:6.2f}% ({correct}/{num_trials})")

    print("="*60)
    return results



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    VOCAB_SIZE = 16000
    HIDDEN_DIM = 256

    TEST_CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192]

    print("Initializing ARIS Language Model Infrastructure...")
    # LogKV backbone (standard config, doc/logkv.md §6.8): fixed log C level
    # decay + phase embedding (levels=2) + multi-head + gated attention.
    # Maps (B, L, HIDDEN_DIM) -> (B, L, HIDDEN_DIM); the wrapper above owns
    # the token embedding / norm / head.
    from logkv import LogKVBlock

    class LogKVBackbone(nn.Module):
        def __init__(self, dim, num_layers=2, num_heads=4, d_ff=512, chunk_size=4):
            super().__init__()
            self.layers = nn.ModuleList([
                LogKVBlock(dim, chunk_size, d_ff, num_heads,
                           phase_emb=True, phase_levels=2, gated_attention=True)
                for _ in range(num_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    backbone = LogKVBackbone(HIDDEN_DIM)
    print(f"Backbone ARIS-Net initialized with {sum(p.numel() for p in backbone.parameters())} parameters.")
    lm_model = LanguageModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, backbone_model=backbone).to(device)

    train_needle_retrieval(
        model=lm_model, 
        vocab_size=VOCAB_SIZE, 
        device=device, 
        steps=300, 
        batch_size=16, 
        h_dim=HIDDEN_DIM
    )

    evaluate_needle_in_a_haystack(
        model=lm_model, 
        vocab_size=VOCAB_SIZE, 
        context_lengths=TEST_CONTEXT_LENGTHS, 
        device=device, 
        num_trials=20
    )