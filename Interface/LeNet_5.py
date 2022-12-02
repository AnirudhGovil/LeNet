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

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

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
if(device.type == 'cpu'):
    model.load_state_dict(torch.load('LeNet_5.pth', map_location='cpu'))
else:
    model.load_state_dict(torch.load('LeNet_5.pth'))
print(predict("images/number.png", model, device))
