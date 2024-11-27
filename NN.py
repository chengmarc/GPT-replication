# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:14:26 2024

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch.nn as nn

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

