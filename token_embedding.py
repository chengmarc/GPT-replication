# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:07:54 2024

@author: uzcheng
"""

import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from importlib.metadata import version

#from tokenizer import tokenizer, raw_text, enc_text
import tiktoken
import torch

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))


# %%
context_size = 10
for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# %%
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# %%
def visualize_batch(dataloader):
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nInput IDs:\n", inputs)
    print("\nTarget IDs:\n", targets)
    
#dataset = GPTDatasetV1(txt=raw_text, tokenizer=tokenizer, max_length=4, stride=4)
#dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=0)

#visualize_batch(dataloader)

# %%

def visualize_embedding(layer):
    input_ids = torch.tensor([2, 3, 0, 1]) 
    print(layer.weight, "\n") # Weight matrix
    print(layer(input_ids))
    # Each row vector is a representation of the corresponding id
    # For example:  the fourth row [ 0.1600, -1.8064, -1.5723] 
    #               is a vector representation of id 3

vocab_size = 4
output_dim = 3
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

visualize_embedding(embedding_layer)

# %%
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
visualize_embedding(token_embedding_layer)

# %%
context_length = 4

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
visualize_embedding(pos_embedding_layer)

# %%
# phrase:               Good morning
# tokens:               36, 48
# token embedding:      [1, 0] = 36, [0, 1] = 48

# position embedding:   [0.5, 0] = at first position
#                       [0, 0.5] = at second position

# final embedding:      [1.5, 0] = token 36 at first position
#                       [0, 1.5] = token 48 at second position
                            
# %%
#inputs, targets = next(iter(dataloader))

#token_embeddings = token_embedding_layer(inputs)
#pos_embeddings = pos_embedding_layer(torch.arange(context_length))

#input_embeddings = token_embeddings + pos_embeddings
#print(input_embeddings.shape)

