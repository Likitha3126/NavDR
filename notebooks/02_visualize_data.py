import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

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

smartphone = pd.read_csv(smartphone_path, encoding="cp1252")
vehicle = pd.read_csv(vehicle_path, encoding="cp1252")

# Remove accidental spaces from column names
smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()

# --------------------------------------------------
# 2. CREATE TIME AXIS
# --------------------------------------------------

time = smartphone["TIME SINCE START (ms)"] / 1000

# Start time from zero
time = time - time.iloc[0]

# --------------------------------------------------
# 3. SPEED COMPARISON
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    smartphone["GPS SPEED (Kmh)"],
    label="Smartphone GPS Speed"
)

plt.plot(
    time,
    vehicle["Velocity (km/hr)"],
    label="Vehicle Velocity"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (km/h)")
plt.title("Smartphone GPS Speed vs Vehicle Velocity")
plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 4. ACCELEROMETER
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    smartphone["ACCELEROMETER X (m/s²)"],
    label="Accelerometer X"
)

plt.plot(
    time,
    smartphone["ACCELEROMETER Y (m/s²)"],
    label="Accelerometer Y"
)

plt.plot(
    time,
    smartphone["ACCELEROMETER Z (m/s²)"],
    label="Accelerometer Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Smartphone Accelerometer Signals")
plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 5. GYROSCOPE
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    smartphone["GYROSCOPE Yaw (rad/s)"],
    label="Yaw"
)

plt.plot(
    time,
    smartphone["GYROSCOPE Pitch (rad/s)"],
    label="Pitch"
)

plt.plot(
    time,
    smartphone["GYROSCOPE Roll (rad/s)"],
    label="Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Smartphone Gyroscope Signals")
plt.legend()
plt.grid()

plt.show()

# --------------------------------------------------
# 6. GPS TRAJECTORY
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.plot(
    smartphone["GPS LONGITUDE (degrees)"],
    smartphone["GPS LATITUDE (degrees)"]
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Smartphone GPS Trajectory")
plt.grid()

plt.show()

# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\nDataset duration:")
print(f"{time.iloc[-1]:.2f} seconds")

print("\nSpeed statistics:")
print(vehicle["Velocity (km/hr)"].describe())

print("\nAccelerometer statistics:")
print(
    smartphone[
        [
            "ACCELEROMETER X (m/s²)",
            "ACCELEROMETER Y (m/s²)",
            "ACCELEROMETER Z (m/s²)"
        ]
    ].describe()
)