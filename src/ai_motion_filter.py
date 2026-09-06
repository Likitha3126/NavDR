import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest


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

smartphone = pd.read_csv(
    smartphone_path,
    encoding="cp1252"
)

smartphone.columns = smartphone.columns.str.strip()


# ============================================================
# 2. TIME
# ============================================================

time = (
    smartphone["TIME SINCE START (ms)"].to_numpy()
    / 1000.0
)

time = time - time[0]


# ============================================================
# 3. IMU SIGNALS
# ============================================================

acc_x = smartphone[
    "ACCELEROMETER X (m/s²)"
].to_numpy()

acc_y = smartphone[
    "ACCELEROMETER Y (m/s²)"
].to_numpy()

acc_z = smartphone[
    "ACCELEROMETER Z (m/s²)"
].to_numpy()


gyro_yaw = smartphone[
    "GYROSCOPE Yaw (rad/s)"
].to_numpy()

gyro_pitch = smartphone[
    "GYROSCOPE Pitch (rad/s)"
].to_numpy()

gyro_roll = smartphone[
    "GYROSCOPE Roll (rad/s)"
].to_numpy()


gravity_x = smartphone[
    "GRAVITY X (m/s²)"
].to_numpy()

gravity_y = smartphone[
    "GRAVITY Y (m/s²)"
].to_numpy()

gravity_z = smartphone[
    "GRAVITY Z (m/s²)"
].to_numpy()


# ============================================================
# 4. REMOVE GRAVITY
# ============================================================

linear_x = acc_x - gravity_x
linear_y = acc_y - gravity_y
linear_z = acc_z - gravity_z


# ============================================================
# 5. BASIC FEATURES
# ============================================================

acc_magnitude = np.sqrt(
    linear_x ** 2
    + linear_y ** 2
    + linear_z ** 2
)

gyro_magnitude = np.sqrt(
    gyro_yaw ** 2
    + gyro_pitch ** 2
    + gyro_roll ** 2
)


# ============================================================
# 6. JERK
# ============================================================

dt = np.gradient(time)

dt[dt <= 0] = 0.01

jerk_x = np.gradient(
    linear_x
) / dt

jerk_y = np.gradient(
    linear_y
) / dt

jerk_z = np.gradient(
    linear_z
) / dt

jerk_magnitude = np.sqrt(
    jerk_x ** 2
    + jerk_y ** 2
    + jerk_z ** 2
)


# ============================================================
# 7. GYROSCOPE CHANGE
# ============================================================

gyro_change = np.sqrt(
    np.gradient(gyro_yaw) ** 2
    + np.gradient(gyro_pitch) ** 2
    + np.gradient(gyro_roll) ** 2
)


# ============================================================
# 8. BUILD ML FEATURES
# ============================================================

features = np.column_stack(
    [
        acc_magnitude,
        gyro_magnitude,
        jerk_magnitude,
        gyro_change,
        np.abs(linear_x),
        np.abs(linear_y),
        np.abs(linear_z),
        np.abs(gyro_yaw),
        np.abs(gyro_pitch),
        np.abs(gyro_roll),
    ]
)


# ============================================================
# 9. REMOVE INVALID VALUES
# ============================================================

valid = np.isfinite(features).all(axis=1)

features_clean = features[valid]

print()
print("==========================================")
print("AI MOTION FILTER")
print("==========================================")

print(
    "Total samples:",
    len(features)
)

print(
    "Valid samples:",
    len(features_clean)
)


# ============================================================
# 10. TRAIN ISOLATION FOREST
# ============================================================

print()
print("Training Isolation Forest...")


model = IsolationForest(
    n_estimators=150,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model.fit(features_clean)


# ============================================================
# 11. PREDICT ANOMALIES
# ============================================================

prediction = np.ones(
    len(features),
    dtype=int
)

prediction[valid] = model.predict(
    features_clean
)


# Isolation Forest:
#
#  1  = normal
# -1  = anomaly


anomaly_mask = (
    prediction == -1
)

normal_mask = (
    prediction == 1
)


# ============================================================
# 12. STATISTICS
# ============================================================

normal_count = np.sum(
    normal_mask
)

anomaly_count = np.sum(
    anomaly_mask
)

anomaly_percentage = (
    anomaly_count
    / len(prediction)
    * 100
)


print()
print(
    "Normal samples:",
    normal_count
)

print(
    "Anomalous samples:",
    anomaly_count
)

print(
    f"Anomaly percentage:"
    f" {anomaly_percentage:.2f}%"
)


# ============================================================
# 13. SHOW DETECTED ANOMALIES
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    acc_magnitude,
    label="Acceleration magnitude"
)

plt.scatter(
    time[anomaly_mask],
    acc_magnitude[anomaly_mask],
    s=8,
    label="AI detected anomaly"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Acceleration magnitude (m/s²)"
)

plt.title(
    "NavDR — AI Motion Anomaly Detection"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 14. BLACKOUT
# ============================================================

blackout_start = 1200

blackout_end = 1260

blackout_mask = (
    (time >= blackout_start)
    &
    (time <= blackout_end)
)

blackout_anomalies = (
    anomaly_mask
    &
    blackout_mask
)

print()
print(
    "Blackout anomalies:",
    np.sum(blackout_anomalies)
)

print(
    f"Blackout anomaly rate:"
    f" {np.sum(blackout_anomalies) / np.sum(blackout_mask) * 100:.2f}%"
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "models/imu_anomaly_model.pkl"
)

print()
print(
    "Model saved:"
)

print(
    "models/imu_anomaly_model.pkl"
)