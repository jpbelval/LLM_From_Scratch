import torch
import torch.nn as nn
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        context_length = attn_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        masked_simple = attn_weights * mask_simple
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums
        context_vec = attn_weights @ values
        torch.manual_seed(123)
        dropout = nn.Dropout(0.5)
        return dropout(masked_simple_norm)