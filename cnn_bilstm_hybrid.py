import torch
from torch.utils.data import Dataset, DataLoader

class CQCCDataset(Dataset):
    def __init__(self, df):
        self.features = df['cqcc'].values
        self.labels = df['label'].values.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # (19, 63)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)  # (1,) <- UWAGA! w nawiasach
        return x, y

import torch.nn as nn
import torch.nn.functional as F

class AudioDeepfakeDetector(nn.Module):
    def __init__(self, feature_dim=19, lstm_units=32, dense_units=64, dropout_rate=0.5):
        super().__init__()

        # --- FEATURE EXTRACTION ---
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=63, out_channels=64, kernel_size=3, padding=1),  # wejście: 63 cechy
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # redukcja wymiaru czasowego
            nn.Dropout(0.3)               # dodatkowy Dropout
        )

        # --- SEQUENCE MODELING ---
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_units,  # mniejsze LSTM
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- ATTENTION ---
        self.attention = nn.Linear(lstm_units * 2, 1)
        self.layer_norm = nn.LayerNorm(1)

        # --- CLASSIFICATION ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units * 2, dense_units),  # mniejsze dense
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 19, 63)
        x = x.permute(0, 2, 1)           # -> (batch, 63, 19)
        x = self.feature_extractor(x)    # -> (batch, 64, 9)
        x = x.permute(0, 2, 1)           # -> (batch, 9, 64)

        lstm_out, _ = self.bilstm(x)     # -> (batch, 9, 64*2)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_weights = self.layer_norm(attn_weights)
        weighted_out = lstm_out * attn_weights

        x = torch.max(weighted_out, dim=1).values  # Global Max Pooling -> (batch, 64*2)
        x = self.classifier(x)                      # -> (batch, 1)
        return x


