import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

if __name__ == "__main__":
    from core.arguments import ModelArgs
    from core.attention import NSEA
else:
    from .core.arguments import ModelArgs
    from .core.attention import NSEA


world_size = 1
rank = 0
block_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

class Block(nn.Module):
    """
    Transformer model for language modeling.
    """
    def __init__(self, args: ModelArgs):
        super(Block, self).__init__()
        self.args = args
        self.norm = nn.RMSNorm(args.dim)
        self.attn = NSEA(args)
        self.ffnn = nn.Linear(args.dim, args.dim)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffnn(self.norm(x))
        return x


class ExpTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        super(ExpTransformer, self).__init__()
        self.args = args
        self.embd = nn.Embedding(args.vocab_size, args.dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = nn.RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, s)
        x = self.embd(x)  # (b, s, d)
        for block in self.blocks:
            x = block(x)  # (b, s, d)
        x = self.norm(x)[:, -1]
        logits = self.head(x)  # (b, s, vocab_size)
        return logits


if __name__ == "__main__":
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    embd = nn.Embedding(args.vocab_size, args.dim)
    exp_transformer = ExpTransformer(args)
    out = exp_transformer(x)
    print(out.shape, out)  # should be (2, vocab_size)