# import required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA, KernelPCA
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def getAccuracy(y_pred, actual):
    # Get accuracy for each class
    accuracy = np.zeros(10)
    for i in range(10):
        accuracy[i] = np.sum(y_pred[actual == i] == i) / np.sum(actual == i)

    loss = np.zeros(10)
    for i in range(10):
        loss[i] = np.sum(y_pred[y_test == i] != i) / np.sum(y_test == i)

    return accuracy*100, loss*100



# Define the transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Get the test data

testset = EMNIST('EMNIST_data/', download=True, train=False, transform=transform, split='letters')

# Create the test loader

testloader = DataLoader(testset, batch_size=64, shuffle=True)

X_test = testset.data.numpy()
y_test = testset.targets.numpy()

# Convert the date to 2D array
X_test = X_test.reshape(X_test.shape[0], -1)

models = ["LinearClassifier.pth", "KNN.pth", "PolynomialClassifier.pth", "RBFNetwork.pth", "SVM.pth"]

for model in models:
    currModel = torch.load(model)
    
    y_pred = currModel.predict(X_test)

    # PLotting accuracy for each class
    accuracy, loss = getAccuracy(y_pred, y_test)
    # Print avg accuracy and loss
    print("Model: ", model)
    print("Avg Accuracy: ", np.mean(accuracy))
    print("Avg Loss: ", np.mean(loss))

    # Plotting accuracy and loss for each class
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(10), loss, color='r')
    plt.title("Loss")
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(10), accuracy, color='g')
    plt.title("Accuracy")
    plt.show()
