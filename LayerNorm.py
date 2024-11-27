# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:51:07 2024

@author: Admin

[1] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.

    Layer Normalization
    https://arxiv.org/pdf/1607.06450
    
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN


# %%
class CustomLayerNorm(NN.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = NN.Parameter(torch.ones(emb_dim))
        self.shift = NN.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class LayerNorm(NN.LayerNorm): pass


# %%
if __name__ == "__main__":
    
    batch_example = torch.randn(1, 5)
    layer = NN.Sequential(NN.Linear(5, 6), NN.ReLU())

    # Before Normalization
    before = layer(batch_example)
    print(f"\n{before}\n")

    mean = before.mean(dim=-1, keepdim=True)
    var = before.var(dim=-1, keepdim=True)

    print("Mean:\t\t", float(mean))
    print("Variance:\t", float(var))

    # After Normalization
    after = (before - mean) / torch.sqrt(var)
    print(f"\n{after}\n")

    mean = after.mean(dim=-1, keepdim=True)
    var = after.var(dim=-1, keepdim=True)

    print("Mean:\t\t", float(mean))
    print("Variance:\t", float(var))

