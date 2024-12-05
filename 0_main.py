# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:31:18 2024

@author: uzcheng
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch
from GPT import model, config, generate
from Tokenizer import tokenizer, raw_text
from DataLoader import create_dataloader
from Typewriter import typewrite


# %%
def text_to_token_ids(text, tokenizer):
    
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


# %% Probability calculation example
input1 = tokenizer.encode("This is an")
input2 = tokenizer.encode("What do you")
inputs = torch.tensor([input1, input2])

target1 = tokenizer.encode("is an example")
target2 = tokenizer.encode("do you mean")
targets = torch.tensor([target1, target2])

with torch.no_grad():
    logits = model(inputs)

text_id1, text_id2 = 0, 1
probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
target_probas_1 = probas[text_id1, [0, 1, 2], targets[text_id1]]
target_probas_2 = probas[text_id2, [0, 1, 2], targets[text_id2]]

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
neg_avg_log_probas = torch.mean(log_probas) * -1
print(f"Total probability: {log_probas}")
print(f"Negative average probability: {neg_avg_log_probas} \n")

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
perplexity = torch.exp(loss)
print(f"Loss: {loss}")
print(f"Perplexity: {perplexity} \n")


# %% Load data
train_ratio = 0.90  # Train/validation ratio
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
valid_data = raw_text[split_idx:]

train_loader = create_dataloader(train_data, batch_size=2,
                                 max_length=config["context_length"], 
                                 stride=config["context_length"],
                                 drop_last=True, shuffle=True, num_workers=0)

valid_loader = create_dataloader(valid_data, batch_size=2,
                                 max_length=config["context_length"], 
                                 stride=config["context_length"],
                                 drop_last=False, shuffle=False, num_workers=0)

train_tokens, valid_tokens = 0, 0
for input_batch, target_batch in train_loader: train_tokens += input_batch.numel()
for input_batch, target_batch in valid_loader: valid_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", valid_tokens)
print("All tokens:", train_tokens + valid_tokens)


# %% Loss calculation
def calc_loss_batch(input_batch, target_batch, model, device):
    
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# %% CUDA acceleration
if torch.cuda.is_available(): 
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")

_ = model.to(device)
print(f"Using {device} device.")


# %%
def print_sample(model, tokenizer, device, start_context):
    
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    typewrite("\n[Sample Text] " + start_context + decoded_text.replace("\n", " ") + "\n\n")
    model.train()
    
def evaluate(model, train_loader, val_loader, device, eval_iter):
    
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        valid_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, valid_loss


# %%
def train(model, train_loader, val_loader, optimizer, device, num_epochs,
          eval_freq, eval_iter, start_context, tokenizer):
    
    train_losses, valid_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step}): "
                      f"Training loss {train_loss:.3f}, Validation loss {valid_loss:.3f}")

        # Print a sample text after each epoch
        print_sample(model, tokenizer, device, start_context)

    return train_losses, valid_losses, track_tokens_seen

    
# %%
num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_losses, valid_losses, tokens_seen = train(
    model, train_loader, valid_loader, optimizer, device, num_epochs, 
    eval_freq=5, eval_iter=5, start_context="It is a sunny day,", tokenizer=tokenizer)

if not os.path.exists('model'):
    os.makedirs('model')
    
torch.save(model.state_dict(), f'./model/GPT2-epoch-{num_epochs}.pt')

