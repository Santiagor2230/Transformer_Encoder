import datetime
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A function to encapsulate the training loop
def train(model, critetion, optimizer, train_loader, valid_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)

  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = 0
    n_train = 0
    for batch in train_loader:
      #move data to GPU
      batch = {k: v.to(device) for k,v in batch.items()}

      #zero the parameter gradients
      optimizer.zero_grad()

      #forward pass
      outputs = model(batch["input_ids"], batch["attention_mask"])
      loss = criterion(outputs, batch["labels"])

      #backward and optimize
      loss.backward()
      optimizer.step()

      train_loss += loss.item()*batch['input_ids'].size(0)
      n_train += batch["input_ids"].size(0)

    # Get average train loss
    train_loss = train_loss / n_train

    model.eval()
    test_loss = 0
    n_test = 0
    for batch in valid_loader:
      batch = {k: v.to(device) for k,v in batch.items()}
      outputs = model(batch["input_ids"], batch["attention_mask"])
      loss = criterion(outputs, batch["labels"])
      test_loss += loss.item()*batch['input_ids'].size(0)
      n_test += batch["input_ids"].size(0)
    test_loss = test_loss / n_test

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss

    dt = datetime.now()- t0
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: { test_loss:.4f}, Duration: {dt}' )

  return train_losses, test_losses
