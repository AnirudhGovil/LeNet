from model import LeNet5
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# Predict the class of a single image using LeNet5

def predict(image_path, model, device):
    # Define the transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Load the image
    image = Image.open(image_path)
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image
    image = image.resize((28, 28))
    # Convert the image to a tensor
    image = transform(image)
    # Add the batch dimension
    image = image.unsqueeze(0)
    # Move the image to the GPU if available
    image = image.to(device)
    # Move the model to the GPU if available
    model.to(device)
    # Predict the class of the image
    output = model(image)
    # Get the predicted class
    _, predicted = torch.max(output.data, 1)
    # Return the predicted class in cpu
    return predicted.cpu().numpy()[0]

# Print the predicted class of the image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the saved model with the best performance
model = LeNet5()
model.load_state_dict(torch.load('LeNet5\model.pth'))
print(predict("LeNet5\images\\number.png", model, device))
