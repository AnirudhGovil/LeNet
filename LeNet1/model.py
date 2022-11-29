import torch

# LeNet-1 model

class LeNet1(torch.nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5)
        self.conv2 = torch.nn.Conv2d(4, 12, 5)
        self.fc = torch.nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features