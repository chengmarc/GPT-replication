# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:20:48 2024

@author: Admin

[1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.

    Language Models are Unsupervised Multitask Learners
    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch, NN
from Tokenizer import tokenizer, get_dictionary
from Embedding import Embedding
from MaskDropout import Dropout
from LayerNorm import CustomLayerNorm
from TransformerBlock import TransformerBlock


# %%
vocab_size = len(get_dictionary(tokenizer, use_custom=False))
                 
config = {
    "vocab_size": vocab_size,
    "context_length": 256,      # Context length
    "emb_dim": 768,             # Embedding dimension
    "att_heads": 12,            # Number of attention heads
    "layers": 12,               # Number of layers
    "dropout": 0.1,             # Dropout rate
    "qkv_bias": False           # Query-Key-Value bias
}


# %% Sanity Check
if __name__ == "__main__":
    
    input_ = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(config)
    output = block(input_)

    print("Input shape:", input_.shape)
    print("Output shape:", output.shape)


# %%
class CustomGPT(NN.Module):
    
    def __init__(self, config):
        super().__init__()
        self.tok_emb = Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = Dropout(config["dropout"])
        
        self.transformer = NN.Sequential(*[TransformerBlock(config) for _ in range(config["layers"])])
        
        self.final_norm = CustomLayerNorm(config["emb_dim"])
        self.out_head = NN.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

# %% Model Summary
model = CustomGPT(config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


# %% Generation Function
def generate(model, idx, max_new_tokens, context_size):    

    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

    return idx_next


# %% Demonstration
if __name__ == "__main__":
    
    example = "This is an example of "
    encoded = tokenizer.encode(example)    
    idx = torch.tensor(encoded).unsqueeze(0)
    
    output = generate(model=model, idx=idx, max_new_tokens=25, context_size=config["context_length"])
    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    
    from Typewriter import typewrite
    typewrite(decoded_text)

