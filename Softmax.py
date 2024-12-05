# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Y. Lecun, L. Bottou, Y. Bengio, P. Haffner.

    Gradient-based learning applied to document recognition
    http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
    
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch


# %%
if __name__ == "__main__":
        
    inputs = torch.rand(6, 3)           # 6 input tokens, each token has 3 dimensions
    query = inputs[1]                   # 2nd input token is the query

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) 
        # dot product (transpose not necessary here since they are 1-dim vectors)
    print("All attention scores for x^2: \n", attn_scores_2) #(w_2i)


# %% 
def normalize(x):
    return x / x.sum()

def naive_softmax(x, dim=0):
    return torch.exp(x) / torch.exp(x).sum(dim)

def softmax(x, dim=0):
    return torch.softmax(x, dim)


# %%
if __name__ == "__main__":
    
    for func in [normalize, naive_softmax, softmax]:        
        attn_weights_2 = func(attn_scores_2)
        print("Attention weights for x^2:", attn_weights_2)
        print("Sum:", attn_weights_2.sum())
        
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += x_i*attn_weights_2[i]
    print("Context vector for x^2:", context_vec_2)

