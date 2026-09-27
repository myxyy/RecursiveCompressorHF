from models.mamba2.configuration import Mamba2Config


class Mamba2LogKVConfig(Mamba2Config):
    model_type = "mamba2_logkv"

    def __init__(self, num_logkv_layers=2, num_heads=8, d_ff=1024, chunk_size=4,
                 gated_attention=True, self_slot=True, conv_kernel_size=0,
                 phase_emb=False, phase_levels=16, learnable_decay=False,
                 kv_norm=False, level_amplify=False, v_norm_only=False, **kwargs):
        super().__init__(**kwargs)
        for name, value in dict(num_logkv_layers=num_logkv_layers, num_heads=num_heads,
                                d_ff=d_ff, chunk_size=chunk_size).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if chunk_size < 2 or self.d_model % num_heads:
            raise ValueError("chunk_size must be >=2 and d_model divisible by num_heads")
        if conv_kernel_size != 0:
            raise ValueError("Hybrid uses Mamba-2's causal convolution; set conv_kernel_size=0")
        for name, value in dict(gated_attention=gated_attention, self_slot=self_slot,
                conv_kernel_size=0, phase_emb=phase_emb, phase_levels=phase_levels,
                learnable_decay=learnable_decay, kv_norm=kv_norm,
                level_amplify=level_amplify, v_norm_only=v_norm_only).items():
            setattr(self, name, value)
