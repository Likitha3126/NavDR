import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ============================================================
# 1. SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# 2. LOAD DATA
# ============================================================

smartphone_path = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

vehicle_path = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\V-Vta1a.csv"
)

smartphone = pd.read_csv(
    smartphone_path,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_path,
    encoding="cp1252"
)

smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()


# ============================================================
# 3. SELECT INPUT FEATURES
# ============================================================

feature_columns = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]

X = smartphone[feature_columns].to_numpy(
    dtype=np.float32
)

y = vehicle["Velocity (km/hr)"].to_numpy(
    dtype=np.float32
)


# ============================================================
# 4. REMOVE INVALID VALUES
# ============================================================

valid = (
    np.isfinite(X).all(axis=1)
    &
    np.isfinite(y)
)

X = X[valid]
y = y[valid]

print("Valid samples:", len(X))


# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(X) * 0.8)

X_train_raw = X[:split_index]
y_train_raw = y[:split_index]

X_test_raw = X[split_index:]
y_test_raw = y[split_index:]

print("Training samples:", len(X_train_raw))
print("Testing samples :", len(X_test_raw))


# ============================================================
# 6. NORMALIZE SENSOR DATA
# ============================================================

scaler = StandardScaler()

X_train_raw = scaler.fit_transform(
    X_train_raw
)

X_test_raw = scaler.transform(
    X_test_raw
)


# ============================================================
# 7. CREATE SEQUENCES
# ============================================================

def create_sequences(X, y, sequence_length):

    sequences = []
    targets = []

    for i in range(
        sequence_length,
        len(X)
    ):

        sequence = X[
            i - sequence_length:i
        ]

        target = y[i]

        sequences.append(sequence)
        targets.append(target)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32)
    )


X_train, y_train = create_sequences(
    X_train_raw,
    y_train_raw,
    SEQUENCE_LENGTH
)

X_test, y_test = create_sequences(
    X_test_raw,
    y_test_raw,
    SEQUENCE_LENGTH
)


print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 8. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 9. DATASET / DATALOADER
# ============================================================

train_dataset = torch.utils.data.TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 10. DEFINE GRU MODEL
# ============================================================

class SpeedGRU(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        speed = self.output(
            last_output
        )

        return speed


model = SpeedGRU().to(DEVICE)


# ============================================================
# 11. LOSS + OPTIMIZER
# ============================================================

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 12. TRAIN
# ============================================================

print("\nTraining GRU...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = loss_function(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# 13. EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_tensor.to(DEVICE)
    )

predictions = (
    predictions
    .cpu()
    .numpy()
    .flatten()
)

actual = y_test


# ============================================================
# 14. METRICS
# ============================================================

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

r2 = r2_score(
    actual,
    predictions
)


print("\n================================")
print("GRU SPEED MODEL RESULTS")
print("================================")

print(
    f"MAE  : {mae:.2f} km/h"
)

print(
    f"RMSE : {rmse:.2f} km/h"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 15. SAVE MODEL + SCALER
# ============================================================

torch.save(
    model.state_dict(),
    r"models\speed_gru.pth"
)

joblib.dump(
    scaler,
    r"models\speed_scaler.pkl"
)

print("\nModel saved:")
print("models\\speed_gru.pth")

print("\nScaler saved:")
print("models\\speed_scaler.pkl")