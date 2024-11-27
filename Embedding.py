# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:07:54 2024

@author: uzcheng
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN
from Tokenizer import tokenizer, raw_text
from DataLoader import create_dataloader


# %%
enc_text = tokenizer.encode("This is an example")
context_size = 3
for i in range(1, context_size+1):
    context, desired = enc_text[:i], enc_text[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


# %%
def visualize_batch(dataloader):
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nInput IDs:\n", inputs)
    print("\nTarget IDs:\n", targets)
    

def visualize_embedding(layer):
    input_ids = torch.tensor([2, 3, 0, 1]) 
    print(layer.weight, "\n") # Weight matrix
    print(layer(input_ids))
    # Each row vector is a representation of the corresponding id
    # For example:  the fourth row [ 0.1600, -1.8064, -1.5723] 
    #               is a vector representation of id 3

class Embedding(NN.Embedding): pass


# %%
if __name__ == "__main__":
    
    dataloader = create_dataloader(txt=raw_text, batch_size=8, shuffle=False, drop_last=True, num_workers=0)
    visualize_batch(dataloader)

    vocab_size, context_length = 4, 4
    output_dim = 3

    token_embedding_layer = Embedding(vocab_size, output_dim)
    visualize_embedding(token_embedding_layer)

    position_embedding_layer = Embedding(context_length, output_dim)
    visualize_embedding(position_embedding_layer)


# %%
# phrase:               Good morning
# tokens:               36, 48
# token embedding:      [1, 0] = 36, [0, 1] = 48

# position embedding:   [0.5, 0] = at first position
#                       [0, 0.5] = at second position

# final embedding:      [1.5, 0] = token 36 at first position
#                       [0, 1.5] = token 48 at second position

