import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 6) 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print(f"input x.shape = {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))  # Input: (3, 32, 32);    Output: (6, 30, 30) -> (6, 15, 15)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (16, 10, 10) -> (16, 5, 5)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # Output: (120, )
        x = F.relu(self.fc2(x)) # Output: (84, )
        x = self.fc3(x) # Output: (10, )
        return x