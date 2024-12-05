# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.

    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN
from Softmax import softmax


# %% Diagonal Mask
class CausalAttentionMask(NN.Module):
    
    def __init__(self, attn_scores, context_length, keys):
        super().__init__()
        self.attn_scores = attn_scores
        self.context_length = context_length
        self.keys = keys
        
        mask_tensor = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer('mask', mask_tensor)
        
    def get_weight(self):
        # attention scores -> softmax -> max diagonals with 0 -> normalize
        attn_weights = softmax(self.attn_scores / self.keys.shape[-1]**0.5, dim=-1)
        attn_weights = attn_weights * self.mask.to(self.attn_scores.device)
        row_sums = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights / row_sums
        return attn_weights
    
class EfficientAttentionMask(NN.Module):
     
    def __init__(self, attn_scores, context_length, keys):
        super().__init__()
        self.attn_scores = attn_scores
        self.context_length = context_length
        self.keys = keys
        
        mask_tensor = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer('mask', mask_tensor)
         
    def get_weight(self):        
        # attention scores -> mask diagonal with -inf -> normalize
        mask_bool = self.mask.bool()[:self.context_length, :self.context_length]  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        self.attn_scores.masked_fill_(mask_bool.to(self.attn_scores.device), -torch.inf)
        attn_weights = softmax(self.attn_scores / self.keys.shape[-1]**0.5, dim=-1)
        return attn_weights


# %%
if __name__ == "__main__":
    
    inputs = torch.rand(6, 3)  
    context_length = inputs.shape[0]
    queries = NN.Parameter(inputs)
    keys = NN.Parameter(inputs)
    
    attn_scores = queries @ keys.T
    print(attn_scores)

    model = CausalAttentionMask(attn_scores, context_length, keys)
    print(model.get_weight())

    model = EfficientAttentionMask(attn_scores, context_length, keys)
    print(model.get_weight())


# %% Dropout
class Dropout(NN.Dropout): pass


# %%
if __name__ == "__main__":
    
    dropout = Dropout(p=0.5)        
    example = torch.rand(6, 6)
    print(dropout(example))

