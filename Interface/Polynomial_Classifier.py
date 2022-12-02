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
from PIL import Image



# Predict the class of a single image using KNN
def predict(image_path, model, device):
    # Define the transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Load the image
    image = Image.open(image_path)
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image
    image = image.resize((28, 28))
    # Convert the image to numpy array
    image = np.array(image)
    # Convert to 1D array
    image = image.reshape(1, -1)

    # Predict the class of the image
    output = model.predict(image)
    return output[0]
    

model = torch.load("PolynomialClassifier.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(predict("images/number.png", model, device))