# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

    Attention Is All You Need
    https://arxiv.org/pdf/1706.03762
    
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN
from MaskDropout import EfficientAttentionMask, Dropout


# %%
class SingleHeadAttention(NN.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        
        self.W_query = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = NN.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Weight mask
        weight_mask = EfficientAttentionMask(attn_scores, num_tokens, keys)
        attn_weights = weight_mask.get_weight()
        
        # Dropout
        dropout = Dropout(p=0.1)
        attn_weights = dropout(attn_weights)
        
        # Context vectors
        context_vec = attn_weights @ values
        
        return context_vec

class ParallelSingleHeadAttention(NN.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = NN.ModuleList(
            [SingleHeadAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# %%
if __name__ == "__main__":
    
    inputs = torch.rand(6, 3)           # 6 input tokens, each token has 3 dimensions
    d_in, d_out = inputs.shape[1], 4    # input dimension = 3, output dimension = 4

    batch = torch.stack((inputs, inputs), dim=0)    # create 2 batches
    batch_size, context_length, d_in = batch.shape
    print("Input shape:\t", batch.shape)          
    # 2 batches of 6 input tokens, each token has 3 dimensions

    model = SingleHeadAttention(d_in, d_out, context_length, 0.0)
    context_vecs = model(batch)
    print("Output shape:\t", context_vecs.shape)         
    # 2 batches of 6 output tokens, each token has 4 dimensions
    
    model = ParallelSingleHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = model(batch)
    print("Output shape:\t", context_vecs.shape)
    # 2 batches of 6 output tokens, each token has [4 x num_heads] dimensions


# %%
class CustomMultiHeadAttention(NN.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = NN.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Weight mask
        weight_mask = EfficientAttentionMask(attn_scores, num_tokens, keys)
        attn_weights = weight_mask.get_weight()
        
        # Dropouts
        dropout = Dropout(p=0.1)
        attn_weights = dropout(attn_weights)
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (attn_weights @ values).transpose(1, 2)         
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class MultiHeadAttention(NN.MultiheadAttention): pass


# %%
if __name__ == "__main__":
    
    model = CustomMultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = model(batch)
    print("Output shape:\t", context_vecs.shape)
    # Achieve the same effect with weight splits

