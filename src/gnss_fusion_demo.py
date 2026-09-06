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
# 2. EXTRACT DATA
# ============================================================

time = (
    smartphone["TIME SINCE START (ms)"].to_numpy()
    / 1000.0
)

time = time - time[0]

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

acc_x = smartphone[
    "ACCELEROMETER X (m/s²)"
].to_numpy()

acc_y = smartphone[
    "ACCELEROMETER Y (m/s²)"
].to_numpy()

gravity_x = smartphone[
    "GRAVITY X (m/s²)"
].to_numpy()

gravity_y = smartphone[
    "GRAVITY Y (m/s²)"
].to_numpy()

gyro_yaw = smartphone[
    "GYROSCOPE Yaw (rad/s)"
].to_numpy()


# ============================================================
# 3. REMOVE GRAVITY
# ============================================================

linear_x = acc_x - gravity_x
linear_y = acc_y - gravity_y


# ============================================================
# 4. GNSS BLACKOUT SETTINGS
# ============================================================

blackout_start = 1200
blackout_duration = 60
blackout_end = blackout_start + blackout_duration

before_start = blackout_start - 60
after_end = blackout_end + 60


# ============================================================
# 5. FIND INDICES
# ============================================================

before_indices = np.where(
    (time >= before_start)
    &
    (time < blackout_start)
)[0]

blackout_indices = np.where(
    (time >= blackout_start)
    &
    (time <= blackout_end)
)[0]

after_indices = np.where(
    (time > blackout_end)
    &
    (time <= after_end)
)[0]


start_index = before_indices[0]


# ============================================================
# 6. LOCAL GPS COORDINATES
# ============================================================

earth_radius = 6371000

reference_lat = latitude[
    start_index
]

reference_lon = longitude[
    start_index
]

gps_north = (
    np.deg2rad(
        latitude - reference_lat
    )
    * earth_radius
)

gps_east = (
    np.deg2rad(
        longitude - reference_lon
    )
    * earth_radius
    * np.cos(
        np.deg2rad(reference_lat)
    )
)


# ============================================================
# 7. INITIAL STATE AT BLACKOUT
# ============================================================

blackout_start_index = blackout_indices[0]

initial_speed = (
    gps_speed[blackout_start_index]
    / 3.6
)

initial_heading = np.deg2rad(
    gps_heading[blackout_start_index]
)

velocity_east = (
    initial_speed
    * np.sin(initial_heading)
)

velocity_north = (
    initial_speed
    * np.cos(initial_heading)
)

current_heading = initial_heading

east = gps_east[
    blackout_start_index
]

north = gps_north[
    blackout_start_index
]


# ============================================================
# 8. DEAD RECKONING DURING BLACKOUT
# ============================================================

dr_east = []
dr_north = []
dr_time = []

previous_time = time[
    blackout_start_index
]

for i in blackout_indices:

    current_time = time[i]

    dt = current_time - previous_time

    previous_time = current_time

    if dt < 0:
        dt = 0

    # Update heading using gyro
    current_heading += (
        gyro_yaw[i] * dt
    )

    # Acceleration in phone frame
    ax = linear_x[i]
    ay = linear_y[i]

    # Rotate into local navigation frame
    acceleration_east = (
        ax * np.sin(current_heading)
        +
        ay * np.cos(current_heading)
    )

    acceleration_north = (
        ax * np.cos(current_heading)
        -
        ay * np.sin(current_heading)
    )

    # Update velocity
    velocity_east += (
        acceleration_east * dt
    )

    velocity_north += (
        acceleration_north * dt
    )

    # Update position
    east += (
        velocity_east * dt
    )

    north += (
        velocity_north * dt
    )

    dr_east.append(east)
    dr_north.append(north)
    dr_time.append(current_time)


dr_east = np.array(dr_east)
dr_north = np.array(dr_north)
dr_time = np.array(dr_time)


# ============================================================
# 9. GNSS RECOVERY
# ============================================================

# At GNSS recovery, calculate the position error.
recovery_index = blackout_indices[-1]

drift_east = (
    dr_east[-1]
    - gps_east[recovery_index]
)

drift_north = (
    dr_north[-1]
    - gps_north[recovery_index]
)

drift_error = np.sqrt(
    drift_east**2
    +
    drift_north**2
)


# ============================================================
# 10. BUILD FUSED TRAJECTORY
# ============================================================

# Before blackout:
# GNSS is trusted.

before_east = gps_east[
    before_indices
]

before_north = gps_north[
    before_indices
]


# During blackout:
# DR is used.

blackout_east = dr_east
blackout_north = dr_north


# After recovery:
# GNSS is trusted again.

after_east = gps_east[
    after_indices
]

after_north = gps_north[
    after_indices
]


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("======================================")
print("NavDR GNSS FUSION DEMONSTRATION")
print("======================================")

print(
    f"GNSS normal period : "
    f"{before_start} → {blackout_start} s"
)

print(
    f"GNSS blackout       : "
    f"{blackout_start} → {blackout_end} s"
)

print(
    f"GNSS recovery       : "
    f"{blackout_end} → {after_end} s"
)

print("\nBlackout results:")

print(
    f"DR drift error at recovery: "
    f"{drift_error:.2f} m"
)


# ============================================================
# 12. PLOT
# ============================================================

plt.figure(figsize=(12, 8))


# ------------------------------------------------------------
# GNSS BEFORE BLACKOUT
# ------------------------------------------------------------

plt.plot(
    before_east,
    before_north,
    label="GNSS Navigation"
)


# ------------------------------------------------------------
# DEAD RECKONING
# ------------------------------------------------------------

plt.plot(
    blackout_east,
    blackout_north,
    label="IMU Dead Reckoning"
)


# ------------------------------------------------------------
# GNSS AFTER RECOVERY
# ------------------------------------------------------------

plt.plot(
    after_east,
    after_north,
    label="GNSS Recovered"
)


# ------------------------------------------------------------
# BLACKOUT START
# ------------------------------------------------------------

plt.scatter(
    gps_east[blackout_start_index],
    gps_north[blackout_start_index],
    label="GNSS LOST"
)


# ------------------------------------------------------------
# GNSS RECOVERY
# ------------------------------------------------------------

plt.scatter(
    gps_east[recovery_index],
    gps_north[recovery_index],
    label="GNSS RESTORED"
)


# ------------------------------------------------------------
# DR RECOVERY POSITION
# ------------------------------------------------------------

plt.scatter(
    dr_east[-1],
    dr_north[-1],
    label="DR Position at Recovery"
)


# ------------------------------------------------------------
# FORMAT
# ------------------------------------------------------------

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "NavDR — GNSS Loss, Dead Reckoning and Recovery"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.show()