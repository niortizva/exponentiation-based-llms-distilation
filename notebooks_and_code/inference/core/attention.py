import torch
from torch import nn

if __name__ == "__main__":
    from arguments import ModelArgs
else:
    from .arguments import ModelArgs


class NSEA(nn.Module):
    """
    NSEA Block
    """
    def __init__(self, args):
        super(NSEA, self).__init__()
        self.Wx = nn.Linear(args.dim , args.n_heads)
        self.Lx = nn.Linear(args.n_heads, args.dim, bias=False)
    
    def exponential_head(self, matrix_product: torch.Tensor) -> torch.Tensor:
        s = torch.eye(matrix_product.size(-2,), matrix_product.size(-1))
        mp1 = matrix_product.clone()
        mp2 = mp1.sum(dim=0, keepdim=True) * mp1
        mp3 = mp2.sum(dim=0, keepdim=True) * mp1
        mp4 = mp3.sum(dim=0, keepdim=True) * mp1
        s = s + mp1 + \
            (1.0 /  2.0) * mp2 + \
            (1.0 /  6.0) * mp3 + \
            (1.0 / 24.0) * mp4
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, s, d)
        Wx = self.Wx(x)  # (b, s, n_heads)
        Wx2 = Wx.sum(dim=0, keepdim=True) * Wx
        Lx = self.Lx(Wx2)
        Ex = self.exponential_head(Lx)
        return Ex



if __name__ == "__main__":
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    emb = nn.Embedding(args.vocab_size, args.dim)
    nsea = NSEA(args)
    x_emb = emb(x)
    out = nsea(x_emb)
    print(out.shape)  # Expected output shape: (2, 128, args.dim)