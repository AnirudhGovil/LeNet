from model import OneFCMNN
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Define the transform to normalize the data

transform = transforms.Compose([transforms.Resize(20), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Create the TestLoader

testset = EMNIST('EMNIST_data/', download=True, train=False, transform=transform, split='letters')

# Test the trained model with the testset

testloader = DataLoader(testset, batch_size=64, shuffle=True)

# Load the saved model with the best performance

model = OneFCMNN()
model.load_state_dict(torch.load('model.pth'))

# Check if the GPU is available 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU if available

model.to(device)

# Define the function to test the model's performance for each class and plot the results

def test_model(model, testloader, device):
    # Define the classes
    classes = ('N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    # Define the number of classes
    n_classes = len(classes)
    # Define the correct and total number of images
    correct = 0
    total = 0
    # Define the list to store the number of correct images for each class
    correct_pred = {classname: 0 for classname in classes}
    # Define the list to store the number of total images for each class
    total_pred = {classname: 0 for classname in classes}
    # Define the list to store the accuracy for each class
    accuracy = {classname: 0 for classname in classes}
    # Define the list to store the loss for each class
    loss = {classname: 0 for classname in classes}
    # Calculate the loss and accuracy for each class
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, predict in zip(labels, predicted):
                if label == predict:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Calculate the accuracy for each class
    for classname in classes:
        accuracy[classname] = 100 * correct_pred[classname] / total_pred[classname]
    # Calculate the loss for each class
    for classname in classes:
        loss[classname] = 100 * (1 - accuracy[classname] / 100)
    # Plot the accuracy for each class
    plt.figure(figsize=(10, 5))
    plt.bar(classes, accuracy.values(), color='g')
    plt.title("Test Accuracy for each class")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.show()
    # Plot the loss for each class
    plt.figure(figsize=(10, 5))
    plt.bar(classes, loss.values(), color='r')
    plt.title("Test Loss for each class")
    plt.xlabel("Class")
    plt.ylabel("Loss")
    plt.show()
    # Print the total accuracy
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    # Print the total loss
    print('Loss of the network on the 10000 test images: %d %%' % (100 - 100 * correct / total))

# Test the model

test_model(model, testloader, device)