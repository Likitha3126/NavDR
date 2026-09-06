import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
# 2. EXTRACT SENSOR DATA
# ============================================================

time = smartphone["TIME SINCE START (ms)"].to_numpy() / 1000.0

# Make time start from zero
time = time - time[0]

acc_x = smartphone["ACCELEROMETER X (m/s²)"].to_numpy()
acc_y = smartphone["ACCELEROMETER Y (m/s²)"].to_numpy()
acc_z = smartphone["ACCELEROMETER Z (m/s²)"].to_numpy()

gravity_x = smartphone["GRAVITY X (m/s²)"].to_numpy()
gravity_y = smartphone["GRAVITY Y (m/s²)"].to_numpy()
gravity_z = smartphone["GRAVITY Z (m/s²)"].to_numpy()

gyro_yaw = smartphone["GYROSCOPE Yaw (rad/s)"].to_numpy()

latitude = smartphone["GPS LATITUDE (degrees)"].to_numpy()
longitude = smartphone["GPS LONGITUDE (degrees)"].to_numpy()

orientation_yaw = smartphone["ORIENTATION (Yaw) (Â°)"].to_numpy()

# ============================================================
# 3. REMOVE GRAVITY
# ============================================================

# Raw accelerometer contains gravity.
# Approximate linear acceleration:

linear_x = acc_x - gravity_x
linear_y = acc_y - gravity_y
linear_z = acc_z - gravity_z


# ============================================================
# 4. CHOOSE GNSS BLACKOUT
# ============================================================

# We simulate a 60-second GNSS outage.
blackout_start = 1200
blackout_duration = 60
blackout_end = blackout_start + blackout_duration

blackout_mask = (
    (time >= blackout_start) &
    (time <= blackout_end)
)

print("GNSS BLACKOUT")
print(f"Start: {blackout_start} seconds")
print(f"End:   {blackout_end} seconds")


# ============================================================
# 5. FIND BLACKOUT START INDEX
# ============================================================

start_index = np.where(blackout_mask)[0][0]
end_index = np.where(blackout_mask)[0][-1]

print(f"Start index: {start_index}")
print(f"End index:   {end_index}")


# ============================================================
# 6. INITIAL POSITION
# ============================================================

start_lat = latitude[start_index]
start_lon = longitude[start_index]

print("\nInitial position:")
print("Latitude :", start_lat)
print("Longitude:", start_lon)


# ============================================================
# 7. INITIAL HEADING
# ============================================================

# Use smartphone orientation only at the moment
# GNSS blackout begins.

heading = np.deg2rad(
    orientation_yaw[start_index]
)

print("Initial heading:", np.rad2deg(heading), "degrees")


# ============================================================
# 8. DEAD RECKONING
# ============================================================

# Local coordinates:
# x = East
# y = North

east = 0.0
north = 0.0

east_positions = []
north_positions = []

current_heading = heading

velocity_east = 0.0
velocity_north = 0.0


for i in range(start_index, end_index + 1):

    if i == start_index:
        dt = 0.0
    else:
        dt = time[i] - time[i - 1]

    # --------------------------------------------------------
    # Update heading using gyroscope
    # --------------------------------------------------------

    current_heading += gyro_yaw[i] * dt

    # --------------------------------------------------------
    # Transform smartphone acceleration into local coordinates
    # --------------------------------------------------------

    # Device X/Y acceleration
    # rotated using estimated heading.

    acceleration_x = linear_x[i]
    acceleration_y = linear_y[i]

    acceleration_east = (
        acceleration_x * np.cos(current_heading)
        - acceleration_y * np.sin(current_heading)
    )

    acceleration_north = (
        acceleration_x * np.sin(current_heading)
        + acceleration_y * np.cos(current_heading)
    )

    # --------------------------------------------------------
    # Integrate acceleration -> velocity
    # --------------------------------------------------------

    velocity_east += acceleration_east * dt
    velocity_north += acceleration_north * dt

    # --------------------------------------------------------
    # Integrate velocity -> position
    # --------------------------------------------------------

    east += velocity_east * dt
    north += velocity_north * dt

    east_positions.append(east)
    north_positions.append(north)


east_positions = np.array(east_positions)
north_positions = np.array(north_positions)


# ============================================================
# 9. CONVERT GPS TO LOCAL METERS
# ============================================================

# Approximate conversion around the starting point.

earth_radius = 6371000

gps_lat = latitude[start_index:end_index + 1]
gps_lon = longitude[start_index:end_index + 1]

gps_north = np.deg2rad(
    gps_lat - start_lat
) * earth_radius

gps_east = (
    np.deg2rad(gps_lon - start_lon)
    * earth_radius
    * np.cos(np.deg2rad(start_lat))
)


# ============================================================
# 10. PLOT GPS VS DEAD RECKONING
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    gps_east,
    gps_north,
    label="GPS Reference"
)

plt.plot(
    east_positions,
    north_positions,
    label="IMU Dead Reckoning"
)

plt.scatter(
    0,
    0,
    label="Blackout Start"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")

plt.title(
    "GNSS Blackout: GPS vs Smartphone Dead Reckoning"
)

plt.legend()
plt.grid()

plt.axis("equal")

plt.show()


# ============================================================
# 11. FINAL ERROR
# ============================================================

final_error = np.sqrt(
    (east_positions[-1] - gps_east[-1]) ** 2
    +
    (north_positions[-1] - gps_north[-1]) ** 2
)

distance_travelled = np.sqrt(
    gps_east[-1] ** 2 +
    gps_north[-1] ** 2
)

print("\n==============================")
print("DEAD RECKONING RESULTS")
print("==============================")

print(
    f"Reference displacement: "
    f"{distance_travelled:.2f} m"
)

print(
    f"Dead reckoning error: "
    f"{final_error:.2f} m"
)

if distance_travelled > 0:

    drift_percentage = (
        final_error / distance_travelled
    ) * 100

    print(
        f"Drift percentage: "
        f"{drift_percentage:.2f}%"
    )