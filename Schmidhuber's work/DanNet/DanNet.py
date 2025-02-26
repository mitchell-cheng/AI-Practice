import torch.nn as nn
import torch.nn.functional as F

class DanNet(nn.Module):
    def __init__(self):
        super(DanNet, self).__init__()

        self.conv1 = nn.Conv2d(2, 100, kernel_size=5)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(100, 100, kernel_size=5)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(100, 100, kernel_size=4)
        self.mp3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(100 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 6)
    def forward(self, x):
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = F.relu(self.mp3(self.conv3(x)))
        x = x.view(-1, 100 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))