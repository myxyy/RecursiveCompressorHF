import torch
import torch.nn as nn

class Compressor(nn.Module):
    def __init__(self):
        super(Compressor, self).__init__()

    def forward(self, q, k, v):
        batch_size, seq_len, d_model = q.size()
        assert k.size() == (batch_size, seq_len, d_model), "Key tensor shape must match query tensor shape"
        assert v.size() == (batch_size, seq_len, d_model), "Value tensor shape must match query tensor shape"

        q_out = q[:, -1, :].unsqueeze(1)  # Take the last query vector and keep the batch dimension
        attention_logits = torch.bmm(q_out, k.transpose(1, 2)) * (d_model ** -0.5)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        k_out = torch.bmm(attention_weights, k)
        v_out = torch.bmm(attention_weights, v)

        return q_out, k_out, v_out

class LogKV(nn.Module):
    def __init__(self, dim, chunk_size):
        super(LogKV, self).__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.lq = nn.Linear(dim, dim, bias=False)
        self.lk = nn.Linear(dim, dim, bias=False)
        self.lv = nn.Linear(dim, dim, bias=False)
        self.compressor = Compressor()

    def forward(self, x):
        chunk_size = self.chunk_size
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, "Input dimension must match the specified dimension"
        q = self.lq(x) # (batch_size, seq_len, d_model)
        k = self.lk(x)
        v = self.lv(x)

        k_list, v_list = self.forward_list_list(q, k, v)
        attention_logits_list = []
        for i in range(len(k_list)):
            batch_size, comp_len, _, d_model = k_list[i].size()
            unit_len = chunk_size ** (i+1)
            comp_len = comp_len + 1
            former_k = torch.cat([k_list[i], torch.zeros(batch_size, 1, chunk_size, d_model, device=k.device)], dim=1) # (batch_size, comp_len, chunk_size, d_model)
            latter_k = torch.cat([torch.zeros(batch_size, 1, chunk_size, d_model, device=k.device), k_list[i]], dim=1) # (batch_size, comp_len, chunk_size, d_model)
            former_kv_mask = (torch.ones(chunk_size, chunk_size, device=k.device).triu())[None, None, :, :].expand(batch_size, comp_len, -1, -1).reshape(batch_size, comp_len * chunk_size, chunk_size)
            latter_kv_mask = 1 - former_kv_mask
            prefix_mask = 1 - torch.ones(comp_len * chunk_size, self.chunk_size, device=k.device).triu()[None, :, :].expand(batch_size, -1, -1)
            prefix_mask_inf = prefix_mask.masked_fill(prefix_mask == 0, float('-inf')).masked_fill(prefix_mask == 1, 0.0)

            former_k_expanded = former_k.unsqueeze(2).expand(-1, -1, unit_len, -1, -1) # (batch_size, comp_len, unit_len, chunk_size, d_model)
            latter_k_expanded = latter_k.unsqueeze(2).expand(-1, -1, unit_len, -1, -1) # (batch_size, comp_len, unit_len, chunk_size, d_model)
            q_pad_len = comp_len * unit_len - seq_len
            q_padded = torch.cat([q, torch.zeros(batch_size, q_pad_len, d_model, device=q.device)], dim=1) if q_pad_len > 0 else q
            q_reshaped = q_padded.view(batch_size, comp_len, unit_len, d_model) # (batch_size, comp_len, unit_len, d_model)
            attention_logits_former = torch.einsum('bculd,bculd->bcull', q_reshaped.unsqueeze(3), former_k_expanded) * former_kv_mask.unsqueeze(2) * (d_model ** -0.5) # (batch_size, comp_len, unit_len, chunk_size)
            attention_logits_latter = torch.einsum('bculd,bculd->bcull', q_reshaped.unsqueeze(3), latter_k_expanded) * latter_kv_mask.unsqueeze(2) * (d_model ** -0.5) # (batch_size, comp_len, unit_len, chunk_size)
            attention_logits = (attention_logits_former + attention_logits_latter) + prefix_mask_inf.unsqueeze(2)
            attention_logits = attention_logits.view(batch_size, comp_len * unit_len, chunk_size) # (batch_size, comp_len * unit_len, chunk_size)
            attention_logits = attention_logits[:, :seq_len, :]  # Remove padding if any
            attention_logits_list.append(attention_logits)

        attention_logits = torch.cat(attention_logits_list, dim=-1) # (batch_size, seq_len, total_kv_len)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        # split attention weights into total_kv_len // chunk_size parts
        attention_weights_list = torch.split(attention_weights, chunk_size, dim=-1)

        v_out_list = []

        for i in range(len(attention_weights_list)):
            attention_weight = attention_weights_list[i] # (batch_size, seq_len, chunk_size)
            former_v = torch.cat([v_list[i], torch.zeros(batch_size, 1, chunk_size, d_model, device=v.device)], dim=1) # (batch_size, comp_len, chunk_size, d_model)
            latter_v = torch.cat([torch.zeros(batch_size, 1, chunk_size, d_model, device=v.device), v_list[i]], dim=1) # (batch_size, comp_len, chunk_size, d_model)
            former_v_expanded = former_v.unsqueeze(2).expand(-1, -1, unit_len, -1, -1) # (batch_size, comp_len, unit_len, chunk_size, d_model)
            latter_v_expanded = latter_v.unsqueeze(2).expand(-1, -1, unit_len, -1, -1) # (batch_size, comp_len, unit_len, chunk_size, d_model)
            former_kv_mask = (torch.ones(chunk_size, chunk_size, device=k.device).triu())[None, None, :, :].expand(batch_size, comp_len, -1, -1).reshape(batch_size, comp_len * chunk_size, chunk_size)
            latter_kv_mask = 1 - former_kv_mask
            attention_weights_former = attention_weight * former_kv_mask[:, :seq_len, :]
            attention_weights_latter = attention_weight * latter_kv_mask[:, :seq_len, :]
            former_v_out = torch.einsum('bsl,bsld->bsd', attention_weights_former, former_v_expanded.view(batch_size, comp_len * unit_len, chunk_size, d_model)[:, :seq_len, :, :]) # (batch_size, seq_len, d_model)
            latter_v_out = torch.einsum('bsl,bsld->bsd', attention_weights_latter, latter_v_expanded.view(batch_size, comp_len * unit_len, chunk_size, d_model)[:, :seq_len, :, :]) # (batch_size, seq_len, d_model)
            v_out = former_v_out + latter_v_out # (batch_size, seq_len, d_model)
            v_out_list.append(v_out)

        v_out = torch.stack(v_out_list, dim=0).sum(dim=0) # (batch_size, seq_len, d_model)
        return v_out

    def forward_list_list(self, q, k, v):
        chunk_size = self.chunk_size
        batch_size, seq_len, d_model = q.size()
        assert k.size() == (batch_size, seq_len, d_model), "Key tensor shape must match query tensor shape"
        assert v.size() == (batch_size, seq_len, d_model), "Value tensor shape must match query tensor shape"

        k_list = []
        v_list = []

        while seq_len > chunk_size:
            pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
            if pad_len > 0:
                q = torch.cat([q, torch.zeros(batch_size, pad_len, d_model, device=q.device)], dim=1)
                k = torch.cat([k, torch.zeros(batch_size, pad_len, d_model, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros(batch_size, pad_len, d_model, device=v.device)], dim=1)
            num_chunks = q.size(1) // chunk_size

            k_list.append(k.reshape(batch_size, num_chunks, chunk_size, d_model))
            v_list.append(v.reshape(batch_size, num_chunks, chunk_size, d_model))

            q_, k_, v_ = self.compressor(
                q.reshape(batch_size * num_chunks, chunk_size, d_model),
                k.reshape(batch_size * num_chunks, chunk_size, d_model),
                v.reshape(batch_size * num_chunks, chunk_size, d_model))
            q_ = q_.reshape(batch_size, num_chunks, d_model)
            k_ = k_.reshape(batch_size, num_chunks, d_model)
            v_ = v_.reshape(batch_size, num_chunks, d_model)

            q = q_
            k = k_
            v = v_
            seq_len = q.size(1)

        return k_list, v_list
