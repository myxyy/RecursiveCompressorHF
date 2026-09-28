from models.logkv.configuration import LogKVConfig
from models.mamba2.configuration import Mamba2Config


class GatedMambaLogKVConfig(LogKVConfig):
    model_type = "mamba2_logkv_gated"

    def __init__(self, num_mamba_layers=2, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1, scan_chunk_size=256,
                 conv_kernel_size=4, gated_attention=True, self_slot=True, **kwargs):
        super().__init__(conv_kernel_size=conv_kernel_size, gated_attention=gated_attention,
                         self_slot=self_slot, **kwargs)
        if conv_kernel_size < 1:
            raise ValueError("Gated variant preserves LogKV's causal convolution (width >=1)")
        for name, value in dict(num_mamba_layers=num_mamba_layers, d_state=d_state,
                d_conv=d_conv, expand=expand, headdim=headdim, ngroups=ngroups,
                scan_chunk_size=scan_chunk_size).items():
            setattr(self, name, value)
        self.mamba_config()  # Validate Mamba dimensions as well as LogKV's.

    def mamba_config(self):
        return Mamba2Config(vocab_size=self.vocab_size, d_model=self.d_model,
                            num_layers=self.num_mamba_layers, d_state=self.d_state,
                            d_conv=self.d_conv, expand=self.expand, headdim=self.headdim,
                            ngroups=self.ngroups, scan_chunk_size=self.scan_chunk_size,
                            pad_token_id=None, bos_token_id=None, eos_token_id=None)
