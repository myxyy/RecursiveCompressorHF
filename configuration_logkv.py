from transformers import PretrainedConfig


class LogKVConfig(PretrainedConfig):
    model_type = "logkv"
    # HF utilities (e.g. GenerationMixin) read num_hidden_layers/hidden_size;
    # map them to our naming.
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
    }

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        num_heads=8,
        d_ff=2048,
        chunk_size=4,
        num_layers=8,
        phase_emb=False,
        phase_levels=16,
        learnable_decay=False,
        gated_attention=False,
        kv_norm=False,
        level_amplify=False,
        v_norm_only=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.phase_emb = phase_emb
        self.phase_levels = phase_levels
        self.learnable_decay = learnable_decay
        self.gated_attention = gated_attention
        self.kv_norm = kv_norm
        self.level_amplify = level_amplify
        self.v_norm_only = v_norm_only
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
