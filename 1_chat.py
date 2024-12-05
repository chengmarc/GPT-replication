# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch
from GPT import model, generate
from Tokenizer import tokenizer
from Typewriter import typewrite


# %%
model_list = [f'./model/{x}' for x in os.listdir('./model')]
if model_list:
    model_list.sort(key=lambda x: os.path.getmtime(x))

    print(f'Loading {model_list[-1]}...')
    model.load_state_dict(torch.load(model_list[-1])) #load latest model
    model.eval()


# %%
while True:
    
    user_input = input("\nUser:\t")
    if user_input == "stop": 
        break
    
    encoded = tokenizer.encode(f"User: {user_input} AI: ")
    idx = torch.tensor(encoded).unsqueeze(0)
    
    token_ids = generate(model=model, idx=idx, max_new_tokens=25, context_size=50)
    
    flat = token_ids.squeeze(0) # remove batch dimension
    decoded_text = "AI:\t" + tokenizer.decode(flat.tolist()) + "\n"
    
    typewrite(decoded_text)

