import torch

# LeNet-1 model

class OneFCMNN(torch.nn.Module):
    def __init__(self):
        super(OneFCMNN, self).__init__()
        self.fc1 = torch.nn.Linear(400, 300)
        self.fc2 = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features