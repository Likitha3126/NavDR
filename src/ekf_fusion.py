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
# 4. CONVERT GPS LAT/LON TO LOCAL METERS
# ============================================================

EARTH_RADIUS = 6371000.0

reference_lat = latitude[0]
reference_lon = longitude[0]

gps_north = (
    np.deg2rad(latitude - reference_lat)
    * EARTH_RADIUS
)

gps_east = (
    np.deg2rad(longitude - reference_lon)
    * EARTH_RADIUS
    * np.cos(np.deg2rad(reference_lat))
)


# ============================================================
# 5. GNSS BLACKOUT
# ============================================================

blackout_start = 1200
blackout_duration = 60
blackout_end = (
    blackout_start
    + blackout_duration
)

before_start = blackout_start - 60
after_end = blackout_end + 60


# ============================================================
# 6. SELECT DEMONSTRATION WINDOW
# ============================================================

window_mask = (
    (time >= before_start)
    &
    (time <= after_end)
)

indices = np.where(window_mask)[0]


# ============================================================
# 7. EKF STATE
#
# x =
# [east_position,
#  north_position,
#  east_velocity,
#  north_velocity,
#  heading]
#
# ============================================================

first_index = indices[0]

initial_speed = (
    gps_speed[first_index] / 3.6
)

initial_heading = np.deg2rad(
    gps_heading[first_index]
)

initial_east_velocity = (
    initial_speed
    * np.sin(initial_heading)
)

initial_north_velocity = (
    initial_speed
    * np.cos(initial_heading)
)


state = np.array([
    gps_east[first_index],
    gps_north[first_index],
    initial_east_velocity,
    initial_north_velocity,
    initial_heading
])


# ============================================================
# 8. INITIAL COVARIANCE
# ============================================================

P = np.diag([
    5.0,       # position east uncertainty
    5.0,       # position north uncertainty
    4.0,       # velocity east uncertainty
    4.0,       # velocity north uncertainty
    0.1        # heading uncertainty
])


# ============================================================
# 9. PROCESS NOISE
# ============================================================

Q = np.diag([
    0.5,
    0.5,
    1.5,
    1.5,
    0.02
])


# ============================================================
# 10. GNSS MEASUREMENT NOISE
# ============================================================

R = np.diag([
    8.0 ** 2,
    8.0 ** 2
])


# ============================================================
# 11. GNSS MEASUREMENT MATRIX
#
# GNSS directly observes:
# east position
# north position
#
# ============================================================

H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])


# ============================================================
# 12. STORAGE
# ============================================================

estimated_east = []
estimated_north = []

estimated_velocity = []
estimated_heading = []

estimated_time = []

error_history = []

mode_history = []


# ============================================================
# 13. EKF LOOP
# ============================================================

previous_time = time[first_index]


for i in indices:

    current_time = time[i]

    dt = current_time - previous_time

    previous_time = current_time

    if dt <= 0:
        dt = 0.01


    # ========================================================
    # PREDICTION STEP
    # ========================================================

    heading = state[4]

    ax = linear_x[i]
    ay = linear_y[i]


    # Transform phone-frame acceleration
    # into navigation frame.

    acceleration_east = (
        ax * np.sin(heading)
        +
        ay * np.cos(heading)
    )

    acceleration_north = (
        ax * np.cos(heading)
        -
        ay * np.sin(heading)
    )


    # Gyroscope updates heading.

    new_heading = (
        heading
        + gyro_yaw[i] * dt
    )


    # Predict position.

    predicted_east = (
        state[0]
        + state[2] * dt
        + 0.5 * acceleration_east * dt * dt
    )

    predicted_north = (
        state[1]
        + state[3] * dt
        + 0.5 * acceleration_north * dt * dt
    )


    # Predict velocity.

    predicted_east_velocity = (
        state[2]
        + acceleration_east * dt
    )

    predicted_north_velocity = (
        state[3]
        + acceleration_north * dt
    )


    state_prediction = np.array([
        predicted_east,
        predicted_north,
        predicted_east_velocity,
        predicted_north_velocity,
        new_heading
    ])


    # ========================================================
    # STATE TRANSITION MATRIX
    # ========================================================

    F = np.eye(5)

    F[0, 2] = dt
    F[1, 3] = dt


    # ========================================================
    # COVARIANCE PREDICTION
    # ========================================================

    P_prediction = (
        F @ P @ F.T
        + Q * max(dt, 0.01)
    )


    # ========================================================
    # GNSS AVAILABILITY
    # ========================================================

    gnss_available = not (
        blackout_start
        <= current_time
        <= blackout_end
    )


    # ========================================================
    # EKF UPDATE
    # ========================================================

    if gnss_available:

        measurement = np.array([
            gps_east[i],
            gps_north[i]
        ])


        # Predicted measurement.

        predicted_measurement = (
            H @ state_prediction
        )


        # Innovation / measurement error.

        innovation = (
            measurement
            - predicted_measurement
        )


        # Innovation covariance.

        S = (
            H
            @ P_prediction
            @ H.T
            + R
        )


        # Kalman gain.

        K = (
            P_prediction
            @ H.T
            @ np.linalg.inv(S)
        )


        # Correct state.

        state = (
            state_prediction
            + K @ innovation
        )


        # Correct covariance.

        identity = np.eye(5)

        P = (
            identity - K @ H
        ) @ P_prediction


        mode = "GNSS + IMU"


    else:

        # GNSS unavailable.
        # Trust inertial prediction.

        state = state_prediction

        P = P_prediction

        mode = "IMU DEAD RECKONING"


    # ========================================================
    # STORE RESULTS
    # ========================================================

    estimated_east.append(state[0])
    estimated_north.append(state[1])

    estimated_velocity.append(
        np.sqrt(
            state[2] ** 2
            +
            state[3] ** 2
        )
        * 3.6
    )

    estimated_heading.append(
        np.rad2deg(state[4])
    )

    estimated_time.append(current_time)

    mode_history.append(mode)


# ============================================================
# 14. CONVERT RESULTS
# ============================================================

estimated_east = np.array(
    estimated_east
)

estimated_north = np.array(
    estimated_north
)

estimated_time = np.array(
    estimated_time
)


# ============================================================
# 15. CALCULATE POSITION ERROR
# ============================================================

reference_east = gps_east[
    indices
]

reference_north = gps_north[
    indices
]

position_error = np.sqrt(
    (
        estimated_east
        - reference_east
    ) ** 2
    +
    (
        estimated_north
        - reference_north
    ) ** 2
)


# ============================================================
# 16. ERROR STATISTICS
# ============================================================

blackout_mask = (
    (estimated_time >= blackout_start)
    &
    (estimated_time <= blackout_end)
)

normal_mask = (
    (estimated_time < blackout_start)
    |
    (estimated_time > blackout_end)
)


blackout_errors = position_error[
    blackout_mask
]

normal_errors = position_error[
    normal_mask
]


print()
print("==========================================")
print("NavDR — EKF GNSS / IMU SENSOR FUSION")
print("==========================================")

print(
    f"GNSS blackout: "
    f"{blackout_start}s → {blackout_end}s"
)

print()

print(
    f"Normal-mode MAE: "
    f"{np.mean(normal_errors):.2f} m"
)

print(
    f"Blackout-mode MAE: "
    f"{np.mean(blackout_errors):.2f} m"
)

print(
    f"Maximum blackout error: "
    f"{np.max(blackout_errors):.2f} m"
)

print(
    f"Error at GNSS recovery: "
    f"{blackout_errors[-1]:.2f} m"
)


# ============================================================
# 17. PLOT TRAJECTORY
# ============================================================

plt.figure(figsize=(12, 8))


plt.plot(
    reference_east,
    reference_north,
    label="GNSS Reference"
)


plt.plot(
    estimated_east,
    estimated_north,
    label="EKF Fused Navigation"
)


# Locate blackout start.

blackout_start_plot = np.where(
    estimated_time >= blackout_start
)[0][0]


plt.scatter(
    estimated_east[blackout_start_plot],
    estimated_north[blackout_start_plot],
    label="GNSS LOST"
)


# Locate recovery.

recovery_plot = np.where(
    estimated_time > blackout_end
)[0][0]


plt.scatter(
    estimated_east[recovery_plot],
    estimated_north[recovery_plot],
    label="GNSS RESTORED"
)


plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "NavDR — EKF GNSS / IMU Sensor Fusion"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.show()


# ============================================================
# 18. ERROR OVER TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    estimated_time,
    position_error,
    label="Position Error"
)

plt.axvline(
    blackout_start,
    linestyle="--",
    label="GNSS Lost"
)

plt.axvline(
    blackout_end,
    linestyle="--",
    label="GNSS Restored"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Position Error (m)"
)

plt.title(
    "NavDR — Position Error During GNSS Blackout"
)

plt.legend()

plt.grid()

plt.show()