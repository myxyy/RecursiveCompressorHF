from transformers import PretrainedConfig


UPSTREAM_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


class Mamba2Config(PretrainedConfig):
    model_type = "mamba2_copying"

    def __init__(self, vocab_size=10, d_model=512, num_layers=2, d_state=128,
                 d_conv=4, expand=2, headdim=64, ngroups=1, scan_chunk_size=256,
                 tie_word_embeddings=True, **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        for name, value in dict(vocab_size=vocab_size, d_model=d_model,
                num_layers=num_layers, d_state=d_state, d_conv=d_conv,
                expand=expand, headdim=headdim, ngroups=ngroups,
                scan_chunk_size=scan_chunk_size).items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if d_model * expand % headdim or (d_model * expand // headdim) % ngroups:
            raise ValueError("Expanded dimension must divide into heads and head groups")
        if d_conv < 2:
            raise ValueError("The upstream unfused convolution path requires d_conv >= 2")
        if scan_chunk_size & (scan_chunk_size - 1):
            raise ValueError("scan_chunk_size must be a power of two")
        self.upstream_revision = UPSTREAM_REVISION

    def upstream_config(self):
        from mamba_ssm.models.config_mamba import MambaConfig
        return MambaConfig(
            d_model=self.d_model, n_layer=self.num_layers, d_intermediate=0,
            vocab_size=self.vocab_size, pad_vocab_size_multiple=1,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
            tie_embeddings=self.tie_word_embeddings,
            ssm_cfg=dict(layer="Mamba2", d_state=self.d_state, d_conv=self.d_conv,
                         expand=self.expand, headdim=self.headdim, ngroups=self.ngroups,
                         chunk_size=self.scan_chunk_size, use_mem_eff_path=False),
        )
