import torch
import torch.nn as nn
from model import CNNModel
from data_handle import get_dataset
from config import EPOCHS


# Instantiate the model
model = CNNModel().to('cuda')

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = torch.optim.Adam(model.parameters())


train_losses = []  # To store training losses
val_losses = []    # To store validation losses
val_accuracies = []  # To store validation accuracies

train_dataloader = get_dataset()
print('strted Training')
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        
        # Move data to GPU
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")
    
torch.save(model.state_dict(), "./model")