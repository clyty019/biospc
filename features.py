#!/usr/bin/env python
# coding: utf-8
"""ResNet feature extraction"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=30):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.res1 = ResBlock1D(hidden_dim)
        self.res2 = ResBlock1D(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.out_proj(x)


def extract_resnet_features(X, out_dim=30, hidden_dim=128, epochs=1000, lr=1e-4):
    """
    Use 1D ResNet for feature extraction and dimensionality reduction on the input matrix.

    Parameters
    ----
    X : np.ndarray
        The original expression matrix (n_cells, n_genes).
    out_dim : int
        The dimension of the reduced feature space.
    hidden_dim : int
        The dimension of the hidden layer.
    epochs : int
        The number of training epochs.
    lr : float
        The learning rate.

    Returns
    ----
    np.ndarray
        The reduced feature matrix (n_cells, out_dim).
     """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = ResNet1D(input_dim=X.shape[1], hidden_dim=hidden_dim, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        feat = model(X_tensor)
        loss = criterion(feat, X_tensor[:, :out_dim])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(X_tensor).numpy()
