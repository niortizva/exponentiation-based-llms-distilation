#!/usr/bin/env python3
"""Benchmark script comparing NSEA, MultiheadAttention, and Hadamard-based exponential attention.

Generates random token indices (to exercise embedding tables of various vocab sizes) and runs
forward passes until `total_tokens` have been processed. Measures per-batch durations and
reports mean and variance of per-token and per-batch timings.

Usage example:
  python benchmark_nsea.py --total-tokens 16384 --seq-len 16 --mini-batch 32
"""
import argparse
import csv
import time
import math
from statistics import mean, pvariance

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, taylor_n=3):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(
            batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, 1, seq_len, seq_len)
            
        Returns:
            attention output and attention weights
        """        
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attention_output, _ = self.scaled_dot_product_attention(
            Q, K, V, mask)
        
        output = self.combine_heads(attention_output) 
        output = self.W_o(output)
        
        return output


class NonSquareExponentialAttention(MultiHeadAttention):
    def __init__(self, *args, taylor_n=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.taylor_n = taylor_n

    @staticmethod
    def pow(C, n):
      C_n = C.clone()
      return C_n.sum(dim=0, keepdim=True)**(n-1) * C_n
    
    def exponential_head(self, matrix_product):
        s = matrix_product ** 0
        for k in range(1, 5):  # Taylor N = 4
            s = s + (1.0 / math.factorial(k)) * pow(matrix_product, k)
        return s
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = self.exponential_head(scores)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights


class HadamardExponentialAttention(NonSquareExponentialAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def pow(C, power):
        return C ** power


def timed_forward(model, x, device, name):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0, out


def run_benchmark(
        vocab_size, device, total_tokens, seq_len, mini_batch, 
        d_model=512, num_heads=16, taylor_n=3):
    # Setup embedding and models
    emb = nn.Embedding(vocab_size, d_model).to(device)

    mha = MultiHeadAttention(
        d_model=d_model, num_heads=num_heads, taylor_n=taylor_n).to(device)
    nsea = NonSquareExponentialAttention(
        d_model=d_model, num_heads=num_heads, taylor_n=taylor_n).to(device)
    hnsea = HadamardExponentialAttention(
        d_model=d_model, num_heads=num_heads, taylor_n=taylor_n).to(device)

    models = [
        ("NonSquareExponential", nsea),
        ("HadamardExponential", hnsea),
        ("MultiHeadAttention", mha),
    ]

    results = []

    for name, model in models:
        durations = []
        tokens_processed = 0
        iterations = 0

        # We'll generate indices in small minibatches to avoid huge memory use
        while tokens_processed < total_tokens:
            batch = min(mini_batch, max(1, (total_tokens - tokens_processed) // seq_len))
            if batch == 0:
                batch = 1

            # sample random token indices
            idx = torch.randint(low=0, high=vocab_size, size=(batch, seq_len), 
                                device=device)
            x = emb(idx)  # (batch, seq_len, p)

            # model expects (batch, seq_len, p) -> forward returns something
            dur, _ = timed_forward(model, x, device, name)

            durations.append(dur)
            tokens_processed += batch * seq_len
            iterations += 1

        # compute per-batch and per-token stats
        per_batch_mean = mean(durations)
        per_batch_var = pvariance(durations)
        per_token_times = [d / (mini_batch * seq_len) for d in durations]
        per_token_mean = mean(per_token_times)
        per_token_var = pvariance(per_token_times)

        results.append({
            "vocab_size": vocab_size,
            "model": name,
            "batches": len(durations),
            "per_batch_mean_s": per_batch_mean,
            "per_batch_variance_s": per_batch_var,
            "per_token_mean_s": per_token_mean,
            "per_token_variance_s": per_token_var,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, default=4096 * 4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--mini-batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" 
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    vocab_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    all_results = []
    print(f'''Running benchmark on device={device}, total_tokens={args.total_tokens}, 
          seq_len={args.seq_len}, mini_batch={args.mini_batch}''')

    for v in vocab_sizes:
        print(f"Benchmarking vocab_size={v}...")
        res = run_benchmark(vocab_size=v, device=device, total_tokens=args.total_tokens, 
                            seq_len=args.seq_len, mini_batch=args.mini_batch)
        all_results.extend(res)

    # write CSV
    keys = ["vocab_size", "model", "batches", "per_batch_mean_s", "per_batch_variance_s", 
            "per_token_mean_s", "per_token_variance_s"]
    with open("computation_efficiency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"Finished. Results written to computation_efficiency.csv")


if __name__ == "__main__":
    main()
