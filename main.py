import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import random

from transformer_encoder_model import Encoder
from training import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#loading the tokenizer
checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#loading the data
raw_datasets = load_dataset("glue", "sst2")

#transformation of the data
def tokenizer_fn(batch):
  return tokenizer(batch["sentence"], truncation=True)

#adjusting data to be tokenized
tokenized_datasets = raw_datasets.map(tokenizer_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#remove columns
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


#tokenized data
train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size = 32,
    collate_fn = data_collator
)

valid_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size = 32,
    collate_fn = data_collator
)

#Intialize Transformer_encoder_model
model = Encoder(
    vocab_size = tokenizer.vocab_size,
    max_len = tokenizer.max_model_input_sizes[checkpoint],
    d_k = 16,
    d_model=64,
    n_heads=4,
    n_layers=2,
    n_classes=2,
    dropout_prob=0.1
)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

#train model
train_losses, test_losses = train(
    model, criterion, optimizer, train_loader, valid_loader, epochs=20
)

#evaluate model
model.eval()
n_correct = 0.
n_total = 0.
for batch in train_loader:
  #move to GPU
  batch = {k: v.to(device) for k,v in batch.items()}

  # Forward pass
  outputs = model(batch["input_ids"], batch["attention_mask"])

  #Get prediction
  # torch.max returns both max and argmax
  _, predictions = torch.max(outputs, 1)

  #update counts
  n_correct += (predictions ==batch["labels"]).sum().item()
  n_total += batch["labels"].shape[0]

train_acc = n_correct / n_total

n_correct = 0
n_total = 0
for batch in valid_loader:

  #move to GPU
  batch = {k:v.to(device) for k,v in batch.items()}

  #forward pass
  outputs = model(batch["input_ids"], batch["attention_mask"])

  #Get predictions
  _, predictions = torch.max(outputs, 1)

  #update counts
  n_correct += (predictions == batch["labels"]).sum().item()
  n_total += batch["labels"].shape[0]

test_acc = n_correct / n_total

print(f" Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")


#random selection of data for predicting
for i in range(3):
  count = 0
  batch_idx = random.randint(0,len(valid_loader)-1)
  for batch in valid_loader:
    idx = random.randint(0,len(batch))

    if count == batch_idx:
      #move to GPU
      batch = {k:v.to(device) for k,v in batch.items()}

      idx = random.randint(0,len(batch))

      end_idx =tokenizer.convert_ids_to_tokens(batch["input_ids"][idx]).index("[SEP]")

      sentence = [token[2:] if token.startswith("#") 
                  else " " +token
                  for token in tokenizer.convert_ids_to_tokens(batch["input_ids"][idx])[1: end_idx]]


        #forward pass
      outputs = model(batch["input_ids"], batch["attention_mask"])

      #Get predictions
      _, predictions = torch.max(outputs, 1)

      print(''.join(sentence)[1:])
      if predictions[idx].item() == 0:
        print("Negative\n")
      else:
        print("Positive\n")
      break

    else:
      count += 1