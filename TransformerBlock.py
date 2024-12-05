# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:11:22 2024

@author: Admin

[1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.

    Language Models are Unsupervised Multitask Learners
    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import NN
from MaskDropout import Dropout
from LayerNorm import CustomLayerNorm
from Activation import CustomGELU
from MultiHeadAttention import CustomMultiHeadAttention


# %%
class FeedForward(NN.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = NN.Sequential(
            NN.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            CustomGELU(),
            NN.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# %%
class TransformerBlock(NN.Module):
    
    def __init__(self, config):
        super().__init__()
        self.att = CustomMultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["att_heads"], 
            dropout=config["dropout"],
            qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = CustomLayerNorm(config["emb_dim"])
        self.norm2 = CustomLayerNorm(config["emb_dim"])
        self.drop_shortcut = Dropout(config["dropout"])

    def forward(self, x):
        
        shortcut = x        # first shortcut
        x = self.norm1(x) 
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x        # second shortcut
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

