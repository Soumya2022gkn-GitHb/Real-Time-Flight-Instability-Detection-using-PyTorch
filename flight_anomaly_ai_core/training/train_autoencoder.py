# =========================================================
# File: training/train_autoencoder.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# Validate PyTorch Installation
# =========================================================

try:

    import torch
    import torch.nn as nn

    from torch.utils.data import (
        DataLoader,
        TensorDataset
    )

except ImportError:

    print("\n========================================")
    print(" ERROR: PyTorch Not Installed ")
    print("========================================")

    print(
        "\nInstall PyTorch using:\n"
    )

    print(
        "pip install torch torchvision torchaudio"
    )

    sys.exit(1)


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42

BATCH_SIZE = 64

LEARNING_RATE = 0.001

NUM_EPOCHS = 30

LATENT_DIM = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(RANDOM_SEED)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = (
    DATASET_DIR
    / "normalized_features.csv"
)

MODEL_FILE = (
    MODELS_DIR
    / "autoencoder.pth"
)

LOSS_HISTORY_FILE = (
    DATASET_DIR
    / "autoencoder_training_loss.csv"
)


# =========================================================
# Create Directories
# =========================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Validate Dataset File
# =========================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" PyTorch Autoencoder Training ")
print("========================================")

print(f"\nLoaded Dataset:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Select Feature Columns
# =========================================================

exclude_columns = [
    "timestamp",
    "anomaly",
    "instability_type"
]

feature_columns = [

    column
    for column in flight_df.columns

    if column not in exclude_columns
]

if len(feature_columns) == 0:

    raise ValueError(
        "\nNo feature columns found."
    )

X = flight_df[
    feature_columns
].values.astype(np.float32)

INPUT_DIM = X.shape[1]

print(f"\nInput Feature Dimension:")
print(INPUT_DIM)


# =========================================================
# Create Tensor Dataset
# =========================================================

tensor_data = torch.tensor(X)

dataset = TensorDataset(
    tensor_data,
    tensor_data
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================================================
# Define Autoencoder
# =========================================================

class Autoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()

        # Encoder

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

        # Decoder

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# =========================================================
# Initialize Model
# =========================================================

model = Autoencoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM
).to(DEVICE)


# =========================================================
# Loss + Optimizer
# =========================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# Training Loop
# =========================================================

loss_history = []

print("\nTraining Started...\n")

for epoch in range(NUM_EPOCHS):

    model.train()

    epoch_loss = 0.0

    for batch_x, _ in dataloader:

        batch_x = batch_x.to(DEVICE)

        # Forward pass

        reconstructed = model(batch_x)

        loss = criterion(
            reconstructed,
            batch_x
        )

        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = (
        epoch_loss / len(dataloader)
    )

    loss_history.append({
        "epoch": epoch + 1,
        "loss": average_loss
    })

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


# =========================================================
# Save Model
# =========================================================

torch.save(
    model.state_dict(),
    MODEL_FILE
)


# =========================================================
# Save Loss History
# =========================================================

loss_df = pd.DataFrame(
    loss_history
)

loss_df.to_csv(
    LOSS_HISTORY_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Autoencoder Training Completed ")
print("========================================")

print(f"\nModel Saved:")
print(MODEL_FILE)

print(f"\nLoss History Saved:")
print(LOSS_HISTORY_FILE)

print(f"\nTraining Device:")
print(DEVICE)

print(f"\nFinal Loss:")
print(loss_history[-1]['loss'])

print("\nSample Loss History:")
print(loss_df.head())
