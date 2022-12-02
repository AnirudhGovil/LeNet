import torch

# LeNet-5 model 

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # self.conv2 = []
        # self.connection_table = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
        #                         [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
        #                         [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
        #                         [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
        #                         [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        #                         [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]]
        # for j in range(16):
        #     num_connection = 0
        #     for i in range(6):
        #         if self.connection_table[i][j] == 1:
        #             num_connection += 1
        #     self.conv2.append(torch.nn.Conv2d(num_connection, 1, 5))
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        # z = torch.zeros(x.shape[0], 16, 10, 10)
        # for j in range(16):
        #     slices = []
        #     for i in range(6):
        #         if self.connection_table[i][j] == 1:
        #             slices.append(x[:, i, :, :])
        #     y = torch.stack(slices, dim=1)
        #     z[:, j, :, :] = self.conv2[j](y).squeeze(1)
        # x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(z), 2)
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


