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
from Softmax import softmax


# %% Simplified Self Attention Example
if __name__ == "__main__":
    
    inputs = torch.rand(6, 3)
    
    # Attention scores
    
    attn_scores = inputs @ inputs.T
    print("Attention scores matrix:\n", attn_scores)

    # Attention weight (normalized)
    
    attn_weights = softmax(attn_scores, dim=-1)
    print("Attention weights matrix:\n", attn_weights)
    print("Sums:", attn_weights.sum(dim=-1))

    # Context vectors
    
    context_vec = attn_weights @ inputs
    print("Context vectors matrix:\n", context_vec)


# %% Trainable Self Attention Example
if __name__ == "__main__":
    
    inputs = torch.rand(6, 3)           # 6 input tokens, each token has 3 dimensions
    d_in, d_out = inputs.shape[1], 2    # input dimension = 3, output dimension = 2

    W_query = NN.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = NN.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = NN.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    keys = inputs @ W_key
    values = inputs @ W_value
    print(keys.shape)               # [number of tokens] x [output dimension]
    print(values.shape)             # [number of tokens] x [output dimension]
  
    # Calculate everything with respect to the second token
    
    x_2 = inputs[1]
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value

    # Attention scores

    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    attn_scores_2 = query_2 @ keys.T
    print("Self attention score of x^2: \n", attn_score_22)
    print("All attention scores for x^2: \n", attn_scores_2)

    # Attention weight (normalized)

    d_k = keys.shape[1]
    attn_weights_2 = softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print("Attention weights for x^2: \n", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # Context vectors

    context_vec_2 = attn_weights_2 @ values
    print("Context vectors for x^2: \n", context_vec_2)


# %% 
class ParameterSelfAttention(NN.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = NN.Parameter(torch.rand(d_in, d_out))
        self.W_key   = NN.Parameter(torch.rand(d_in, d_out))
        self.W_value = NN.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

class LinearSelfAttention(NN.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = NN.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = NN.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


# %% 
if __name__ == "__main__":
    
    model = ParameterSelfAttention(d_in, d_out)
    print(model(inputs))

    model = LinearSelfAttention(d_in, d_out)
    print(model(inputs))

