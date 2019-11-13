from torch import nn
import torch.nn.functional as F

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()
        self.name = "Sound"
        self.conv1 = nn.Conv1d(40, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(10, 5, 5)
        self.fc1 = nn.Linear(2 * 428, 50)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2 * 428)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

