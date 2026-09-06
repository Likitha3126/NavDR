import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ============================================================
# 1. LOAD DATA
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
# 2. SELECT SENSOR SIGNALS
# ============================================================

sensor_columns = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]

sensors = smartphone[sensor_columns].copy()

target = vehicle["Velocity (km/hr)"].copy()


# ============================================================
# 3. CREATE TEMPORAL FEATURES
# ============================================================

print("Creating temporal features...")

features = pd.DataFrame(index=sensors.index)

window = 20


for column in sensor_columns:

    signal = sensors[column]

    # Current value
    features[column + "_current"] = signal

    # Rolling mean
    features[column + "_mean"] = (
        signal.rolling(window).mean()
    )

    # Rolling standard deviation
    features[column + "_std"] = (
        signal.rolling(window).std()
    )

    # Rolling minimum
    features[column + "_min"] = (
        signal.rolling(window).min()
    )

    # Rolling maximum
    features[column + "_max"] = (
        signal.rolling(window).max()
    )


# ============================================================
# 4. ADD ACCELERATION MAGNITUDE
# ============================================================

features["acceleration_magnitude"] = np.sqrt(
    sensors["ACCELEROMETER X (m/s²)"] ** 2
    +
    sensors["ACCELEROMETER Y (m/s²)"] ** 2
    +
    sensors["ACCELEROMETER Z (m/s²)"] ** 2
)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

data = pd.concat(
    [features, target.rename("target_speed")],
    axis=1
)

data = data.dropna()

print("Final training dataset:", data.shape)


# ============================================================
# 6. TIME-BASED SPLIT
# ============================================================

# IMPORTANT:
# We split chronologically instead of randomly.
# This gives us a more realistic evaluation.

split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]


X_train = train_data.drop(
    columns=["target_speed"]
)

y_train = train_data["target_speed"]

X_test = test_data.drop(
    columns=["target_speed"]
)

y_test = test_data["target_speed"]


print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining temporal Random Forest...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. PREDICT
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 9. EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n================================")
print("TEMPORAL AI SPEED MODEL RESULTS")
print("================================")

print(f"MAE  : {mae:.2f} km/h")
print(f"RMSE : {rmse:.2f} km/h")
print(f"R²   : {r2:.4f}")


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = r"models\temporal_speed_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)