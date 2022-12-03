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

class TwoFCMNN(torch.nn.Module):
    def __init__(self):
        super(TwoFCMNN, self).__init__()
        self.fc1 = torch.nn.Linear(784, 1000)
        self.fc2 = torch.nn.Linear(1000, 150)
        self.fc3 = torch.nn.Linear(150, 27)

    def forward(self, x):
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
     # Flip the image
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # Rotate the image 90 degrees anticlockwise
    image = image.rotate(90)
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
model = TwoFCMNN()
if(device.type == 'cpu'):
    model.load_state_dict(torch.load('Two_Hidden_Layer_FCMNN.pth', map_location='cpu'))
else:
    model.load_state_dict(torch.load('Two_Hidden_Layer_FCMNN.pth'))
print(predict("images/number.png", model, device))
