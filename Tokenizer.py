# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Philip Gage.

    A new algorithm for data compression (Byte Pair Encoding)
    https://dl.acm.org/doi/abs/10.5555/177910.177914

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import re, tiktoken


# %%
file_path = "harrypotter.txt"
with open(os.path.join(script_path ,file_path), "r", encoding="utf-8") as file:
    raw_text = file.read()


# %%
def get_vocabulary(raw_text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(list(set(preprocessed)))    
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_words)}
    return vocab


# %%
class CustomTokenizer:
    
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

custom_tokenizer = CustomTokenizer(get_vocabulary(raw_text))

tokenizer = tiktoken.get_encoding("gpt2")

    
# %%
def get_dictionary(tokenizer, use_custom=False):
    
    token_dict = {}
    
    if use_custom:        
        for token_id in range(len(tokenizer.str_to_int)):
            try: token_dict[token_id] = tokenizer.decode([token_id])
            except: print(f"Skipping token ID {token_id}")
            
    else:        
        for token_id in range(tokenizer.n_vocab):
            try: token_dict[token_id] = tokenizer.decode([token_id])
            except: print(f"Skipping token ID {token_id}")
            
    return token_dict


# %%
if __name__ == "__main__":

    dict1 = get_dictionary(custom_tokenizer, use_custom=True)
    dict2 = get_dictionary(tokenizer, use_custom=False)

