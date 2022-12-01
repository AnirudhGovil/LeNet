import torch

# LeNet-4 model

class LeNet4(torch.nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(4, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 27)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
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