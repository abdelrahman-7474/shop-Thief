import torch 
import torch.nn as nn

class VideoModel(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout=0.3):
        super(VideoModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),


            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Global Average Pooling to reduce [B, 512, 7, 7] → [B, 512]
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ----- GRU for temporal modeling -----
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # ----- Fully Connected (Classification) -----
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, frames, 3, 224, 224]
        B, frames, C, H, W = x.shape
        cnn_features = []

        for f_idx in range(frames):
            frame = x[:, f_idx, :, :, :]         # [B, 3, 224, 224]
            f = self.cnn(frame)                  # [B, 512, 7, 7]
            f = f.view(B, -1)                    # [B, 512]
            cnn_features.append(f)

        # Stack all frames → [B, frames, 512]
        feats = torch.stack(cnn_features, dim=1)

        # GRU output
        out, _ = self.gru(feats)                 # [B, frames, hidden_size]
        out = out[:, -1, :]                      # take last frame output
        out = self.fc(out)                       # [B, 1]

        return out
