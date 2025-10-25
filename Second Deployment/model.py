# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def forward_features(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            in_features = x.size(1)
            self.fc1 = nn.Linear(in_features, 512).to(x.device)
            self.fc2 = nn.Linear(512, self.num_classes).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(weights_path: str, device: str = "cpu"):
    model = CNN3D(num_classes=2).to(device)
    # strict=False is helpful if your lazy FC init differs between sessions
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model