# importing the required modules
from classifiers import Classifiers
from torchvision.datasets import MNIST
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


# Define the transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Get the training data

trainset = MNIST('MNIST_data/', download=True, train=True, transform=transform)

# Create the training loader

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

X_train = trainset.data.numpy()
y_train = trainset.targets.numpy()

# Convert the date to 2D array
X_train = X_train.reshape(X_train.shape[0], -1)

# Get the test data

testset = MNIST('MNIST_data/', download=True, train=False, transform=transform)

# Create the test loader

testloader = DataLoader(testset, batch_size=64, shuffle=True)

X_test = testset.data.numpy()
y_test = testset.targets.numpy()

# Convert the date to 2D array
X_test = X_test.reshape(X_test.shape[0], -1)


multClassifier = Classifiers(X_train, y_train, X_test, y_test)

# C1: Linear Classifier
model = multClassifier.LinearClassifier(X_train[:10000, :], y_train[:10000])
torch.save(model, "LinearClassifier.pth")

# # C2: Baseline KNN
model = multClassifier.KNN()
torch.save(model, "KNN.pth")

# C3: Principal Component Analysis (PCA) and Polynomial Classifier
pca = PCA(n_components=40)
X_train_pca = pca.fit_transform(X_train[:10000, :])

model = multClassifier.PolynomialClassifier(X_train_pca, y_train[:10000])
torch.save(model, "PolynomialClassifier.pth")

# # C4: Radial Basis Function (RBF) Network
model = multClassifier.RBFNetwork(X_train[:5000, :], y_train[:5000])
torch.save(model, "RBFNetwork.pth")

# # C11: SVM
model = multClassifier.SVM(X_train[:30000, :], y_train[:30000]) 
torch.save(model, "SVM.pth")

