from transformers import PretrainedConfig


class RecursiveCompressorConfig(PretrainedConfig):
    model_type = "recursive_compressor"
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
        chunk_size=8,
        compress_size=4,
        retrieve_size=4,
        num_layers=8,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.chunk_size = chunk_size
        self.compress_size = compress_size
        self.retrieve_size = retrieve_size
        self.num_layers = num_layers
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
