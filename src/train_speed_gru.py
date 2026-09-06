import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"data\Synchronised V abd S datasets\Synchronised V abd S datasets\Categorised IOVNB Dataset\Vta (Driver E)\Vta01a\S-Vta1a.csv"

MODEL_PATH = r"models\speed_gru_v2.pth"
FEATURE_SCALER_PATH = r"models\speed_feature_scaler_v2.pkl"
TARGET_SCALER_PATH = r"models\speed_target_scaler_v2.pkl"

SEQ_LEN = 20

EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.80


# ============================================================
# MODEL
# ============================================================

class SpeedGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        out, _ = self.gru(x)

        last = out[:, -1, :]

        return self.output(last).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

print("\n======================================")
print(" NavDR — GRU Speed Model V2")
print("======================================\n")

df = pd.read_csv(
    CSV_PATH,
    encoding="cp1252"
)

df.columns = df.columns.str.strip()

print("Dataset:", df.shape)


# ============================================================
# FEATURES
# ============================================================

feature_cols = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]

target_col = "GPS SPEED (Kmh)"


X_raw = df[feature_cols].values.astype(np.float32)

y_raw = df[target_col].values.astype(np.float32)


# ============================================================
# CLEAN
# ============================================================

X_raw = np.nan_to_num(X_raw)

y_raw = np.nan_to_num(y_raw)

y_raw = np.maximum(y_raw, 0)

print(
    "Speed range:",
    round(float(y_raw.min()), 2),
    "to",
    round(float(y_raw.max()), 2),
    "km/h"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split = int(len(X_raw) * TRAIN_RATIO)

X_train_raw = X_raw[:split]

X_test_raw = X_raw[split:]

y_train_raw = y_raw[:split]

y_test_raw = y_raw[split:]


# ============================================================
# FEATURE SCALER
# ============================================================

feature_scaler = StandardScaler()

feature_scaler.fit(X_train_raw)

X_scaled = feature_scaler.transform(X_raw)


# ============================================================
# TARGET SCALER
# ============================================================

target_scaler = StandardScaler()

target_scaler.fit(
    y_train_raw.reshape(-1, 1)
)

y_scaled = target_scaler.transform(
    y_raw.reshape(-1, 1)
).ravel()


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(X, y, start, end):

    sequences = []
    targets = []
    indices = []

    start = max(
        start,
        SEQ_LEN - 1
    )

    for i in range(start, end):

        sequence = X[
            i - SEQ_LEN + 1:i + 1
        ]

        sequences.append(sequence)

        targets.append(y[i])

        indices.append(i)

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            targets,
            dtype=np.float32
        ),
        np.asarray(indices)
    )


X_train, y_train, train_indices = create_sequences(
    X_scaled,
    y_scaled,
    0,
    split
)

X_test, y_test, test_indices = create_sequences(
    X_scaled,
    y_scaled,
    split,
    len(X_scaled)
)


print("\nTraining sequences:", X_train.shape)

print("Testing sequences :", X_test.shape)


# ============================================================
# MODEL
# ============================================================

device = torch.device("cpu")

model = SpeedGRU().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# ============================================================
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train)

y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)

y_test_tensor = torch.tensor(y_test)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining GRU...\n")

for epoch in range(EPOCHS):

    model.train()

    permutation = torch.randperm(
        len(X_train_tensor)
    )

    total_loss = 0

    for start in range(
        0,
        len(X_train_tensor),
        BATCH_SIZE
    ):

        indices = permutation[
            start:start + BATCH_SIZE
        ]

        xb = X_train_tensor[indices]

        yb = y_train_tensor[indices]

        optimizer.zero_grad()

        prediction = model(xb)

        loss = criterion(
            prediction,
            yb
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        max(
            1,
            len(X_train_tensor) // BATCH_SIZE
        )
    )

    if (
        epoch == 0
        or
        (epoch + 1) % 5 == 0
    ):

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {avg_loss:.5f}"
        )


# ============================================================
# EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).numpy()


# Convert back to km/h
predictions = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).ravel()

actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()


predictions = np.maximum(
    predictions,
    0
)

predictions = np.clip(
    predictions,
    0,
    150
)


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


print("\n======================================")
print(" MODEL RESULTS")
print("======================================")

print(
    f"MAE  : {mae:.2f} km/h"
)

print(
    f"RMSE : {rmse:.2f} km/h"
)

print(
    f"R²   : {r2:.4f}"
)

print("======================================\n")


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

joblib.dump(
    feature_scaler,
    FEATURE_SCALER_PATH
)

joblib.dump(
    target_scaler,
    TARGET_SCALER_PATH
)


print("Saved:")
print(MODEL_PATH)
print(FEATURE_SCALER_PATH)
print(TARGET_SCALER_PATH)

print("\nTraining complete.")