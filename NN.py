# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch.nn as nn


# %%
class Module(nn.Module): pass

class ModuleList(nn.ModuleList): pass

class Sequential(nn.Sequential): pass

class Parameter(nn.Parameter): pass

class Linear(nn.Linear): pass

class Embedding(nn.Embedding): pass

class Dropout(nn.Dropout): pass

class LayerNorm(nn.LayerNorm): pass

class MultiheadAttention(nn.MultiheadAttention): pass

class ReLU(nn.ReLU): pass

class GELU(nn.GELU): pass


# %%
"""
───0_main
    │
    ├───Typewriter
    ├───Tokenizer
    ├───DataLoader
    └───GPT
        │
        ├───Tokenizer
        ├───Embedding
        │   │
        │   ├───Tokenizer
        │   └───DataLoader
        │    
        ├───MaskDropout
        │   │
        │   └───Softmax
        │    
        ├───TransformerBlock
        │   │
        │   ├───MultiHeadAttention
        │   │   │
        │   │   └───MaskDropout
        │   │       │
        │   │       └───Softmax
        │   │
        │   ├───Activation
        │   ├───LayerNorm
        │   └───MaskDropout            
        │       │
        │       └───Softmax
        │
        └───LayerNorm
───1_chat

Note 1: SelfAttention is not used
Note 2: NN contains selected class from torch.nn
"""

