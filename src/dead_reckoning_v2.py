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
# 2. SENSOR DATA
# ============================================================

time = (
    smartphone["TIME SINCE START (ms)"].to_numpy()
    / 1000.0
)

time = time - time[0]

acc_x = smartphone[
    "ACCELEROMETER X (m/s²)"
].to_numpy()

acc_y = smartphone[
    "ACCELEROMETER Y (m/s²)"
].to_numpy()

acc_z = smartphone[
    "ACCELEROMETER Z (m/s²)"
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

gyro_yaw = smartphone[
    "GYROSCOPE Yaw (rad/s)"
].to_numpy()

latitude = smartphone[
    "GPS LATITUDE (degrees)"
].to_numpy()

longitude = smartphone[
    "GPS LONGITUDE (degrees)"
].to_numpy()

gps_speed = smartphone[
    "GPS SPEED (Kmh)"
].to_numpy()

gps_heading = smartphone[
    "GPS ORIENTATION (Â°)"
].to_numpy()


# ============================================================
# 3. REMOVE GRAVITY
# ============================================================

linear_x = acc_x - gravity_x
linear_y = acc_y - gravity_y
linear_z = acc_z - gravity_z


# ============================================================
# 4. GNSS BLACKOUT
# ============================================================

blackout_start = 1200
blackout_duration = 60
blackout_end = blackout_start + blackout_duration

mask = (
    (time >= blackout_start)
    &
    (time <= blackout_end)
)

start_index = np.where(mask)[0][0]
end_index = np.where(mask)[0][-1]

print("================================")
print("GNSS BLACKOUT")
print("================================")

print(
    f"Start: {blackout_start} seconds"
)

print(
    f"End:   {blackout_end} seconds"
)


# ============================================================
# 5. INITIAL STATE FROM GNSS
# ============================================================

start_lat = latitude[start_index]
start_lon = longitude[start_index]

# Convert km/h -> m/s
initial_speed = (
    gps_speed[start_index] / 3.6
)

initial_heading = np.deg2rad(
    gps_heading[start_index]
)

print("\nInitial GNSS state:")

print(
    f"Latitude : {start_lat:.6f}"
)

print(
    f"Longitude: {start_lon:.6f}"
)

print(
    f"Speed    : {gps_speed[start_index]:.2f} km/h"
)

print(
    f"Heading  : {gps_heading[start_index]:.2f} degrees"
)


# ============================================================
# 6. INITIAL VELOCITY
# ============================================================

velocity_east = (
    initial_speed
    * np.sin(initial_heading)
)

velocity_north = (
    initial_speed
    * np.cos(initial_heading)
)


# ============================================================
# 7. INITIAL POSITION
# ============================================================

east = 0.0
north = 0.0

current_heading = initial_heading

east_positions = []
north_positions = []


# ============================================================
# 8. DEAD RECKONING
# ============================================================

for i in range(
    start_index,
    end_index + 1
):

    if i == start_index:

        dt = 0.0

    else:

        dt = (
            time[i]
            - time[i - 1]
        )

    # --------------------------------------------------------
    # Update heading using gyro
    # --------------------------------------------------------

    current_heading += (
        gyro_yaw[i] * dt
    )

    # --------------------------------------------------------
    # Smartphone acceleration
    # --------------------------------------------------------

    acceleration_x = linear_x[i]
    acceleration_y = linear_y[i]

    # --------------------------------------------------------
    # Rotate acceleration
    # --------------------------------------------------------

    acceleration_east = (
        acceleration_x
        * np.sin(current_heading)
        +
        acceleration_y
        * np.cos(current_heading)
    )

    acceleration_north = (
        acceleration_x
        * np.cos(current_heading)
        -
        acceleration_y
        * np.sin(current_heading)
    )

    # --------------------------------------------------------
    # Update velocity
    # --------------------------------------------------------

    velocity_east += (
        acceleration_east * dt
    )

    velocity_north += (
        acceleration_north * dt
    )

    # --------------------------------------------------------
    # Update position
    # --------------------------------------------------------

    east += (
        velocity_east * dt
    )

    north += (
        velocity_north * dt
    )

    east_positions.append(east)
    north_positions.append(north)


east_positions = np.array(
    east_positions
)

north_positions = np.array(
    north_positions
)


# ============================================================
# 9. CONVERT GPS TO LOCAL METERS
# ============================================================

earth_radius = 6371000

gps_lat = latitude[
    start_index:end_index + 1
]

gps_lon = longitude[
    start_index:end_index + 1
]

gps_north = (
    np.deg2rad(
        gps_lat - start_lat
    )
    * earth_radius
)

gps_east = (
    np.deg2rad(
        gps_lon - start_lon
    )
    * earth_radius
    * np.cos(
        np.deg2rad(start_lat)
    )
)


# ============================================================
# 10. FINAL ERROR
# ============================================================

final_error = np.sqrt(
    (
        east_positions[-1]
        - gps_east[-1]
    ) ** 2
    +
    (
        north_positions[-1]
        - gps_north[-1]
    ) ** 2
)

reference_distance = np.sqrt(
    gps_east[-1] ** 2
    +
    gps_north[-1] ** 2
)

drift_percentage = (
    final_error
    / reference_distance
) * 100


# ============================================================
# 11. RESULTS
# ============================================================

print("\n================================")
print("DEAD RECKONING V2 RESULTS")
print("================================")

print(
    f"Reference displacement: "
    f"{reference_distance:.2f} m"
)

print(
    f"Final DR error: "
    f"{final_error:.2f} m"
)

print(
    f"Drift: "
    f"{drift_percentage:.2f}%"
)


# ============================================================
# 12. PLOT
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
    label="GNSS Blackout Start"
)

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "GNSS Blackout — Improved Dead Reckoning"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.show()