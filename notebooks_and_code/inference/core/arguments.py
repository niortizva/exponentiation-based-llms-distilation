from dataclasses import dataclass


@dataclass
class ModelArgs:
    vocab_size: int = 50257  # Default vocab size, will be updated after tokenizer is loaded
    dim: int = 760
    n_heads: int = 5
    inter_dim: int = 2048
    n_layers: int = 3
    original_seq_len: int = 1024
    max_batch_size: int = 8
