# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:52:49 2024

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN


# %%
class CustomGELU(NN.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 *x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))*(x + 0.044715 * torch.pow(x, 3))))
    
class GELU(NN.GELU): pass

class RELU(NN.ReLU): pass


# %%
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    gelu, relu = GELU(), RELU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

