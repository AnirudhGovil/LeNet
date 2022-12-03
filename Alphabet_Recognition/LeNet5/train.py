# Import the EMNIST dataset

from torchvision.datasets import EMNIST

# Import the transforms

from torchvision import transforms

# Import the DataLoader

from torch.utils.data import DataLoader

# Import the Adam optimizer

from torch.optim import Adam

# Import the CrossEntropyLoss

from torch.nn import CrossEntropyLoss

# Import the torch.nn.functional

import torch.nn.functional as F

# Import the torch

import torch

# Import the matplotlib

import matplotlib.pyplot as plt

# Import the numpy

import numpy as np

# Import the time

import time

# Import the os

import os

import torch

# LeNet-5 model 

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 27)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Run the code on the GPU if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Download and load the training data

trainset = EMNIST('EMNIST_data/', download=True, train=True, transform=transform, split='letters')

# Train the model with the trainset

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Define the model, optimizer and the loss function

model = LeNet5().to(device)

optimizer = Adam(model.parameters(), lr=0.001)

criterion = CrossEntropyLoss()

# Define the number of epochs to train the model

n_epochs = 8

# Lists to keep track of loss and accuracy

train_losses, test_losses = [], []

# For each epoch

for epoch in range(n_epochs):
    
        # Monitor the training loss
    
        train_loss = 0.0
    
        # Train the model
    
        model.train()
    
        # For each batch in the training set
    
        for data, target in trainloader:
    
            # Move tensors to GPU if CUDA is available
    
            data, target = data.to(device), target.to(device)
    
            # Clear the gradients of all optimized variables
    
            optimizer.zero_grad()
    
            # Forward pass: compute predicted outputs by passing inputs to the model
    
            output = model(data)
    
            # Calculate the batch loss
    
            loss = criterion(output, target)
    
            # Backward pass: compute gradient of the loss with respect to model parameters
    
            loss.backward()
    
            # Perform a single optimization step (parameter update)
    
            optimizer.step()
    
            # Update training loss
    
            train_loss += loss.item() * data.size(0)
    
        # Calculate the average loss over an epoch
    
        train_loss = train_loss / len(trainloader.dataset)
    
        # Print training statistics
    
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    
        # Append training statistics
    
        train_losses.append(train_loss)

# Plot the training loss

plt.plot(train_losses, label='Training loss')

plt.legend(frameon=False)

plt.show()

# Save the model

torch.save(model.state_dict(), 'model.pth')



