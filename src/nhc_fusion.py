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
# 3. LINEAR ACCELERATION
# ============================================================

linear_x = acc_x - gravity_x
linear_y = acc_y - gravity_y


# ============================================================
# 4. GPS → LOCAL METERS
# ============================================================

EARTH_RADIUS = 6371000.0

reference_lat = latitude[0]

reference_lon = longitude[0]

gps_north = (
    np.deg2rad(
        latitude - reference_lat
    )
    * EARTH_RADIUS
)

gps_east = (
    np.deg2rad(
        longitude - reference_lon
    )
    * EARTH_RADIUS
    * np.cos(
        np.deg2rad(reference_lat)
    )
)


# ============================================================
# 5. BLACKOUT
# ============================================================

blackout_start = 1200

blackout_duration = 60

blackout_end = (
    blackout_start
    + blackout_duration
)

before_start = (
    blackout_start - 60
)

after_end = (
    blackout_end + 60
)


# ============================================================
# 6. WINDOW
# ============================================================

window_mask = (
    (time >= before_start)
    &
    (time <= after_end)
)

indices = np.where(
    window_mask
)[0]


# ============================================================
# 7. INITIAL STATE
# ============================================================

first_index = indices[0]

initial_speed = (
    gps_speed[first_index]
    / 3.6
)

initial_heading = np.deg2rad(
    gps_heading[first_index]
)

velocity_east = (
    initial_speed
    * np.sin(initial_heading)
)

velocity_north = (
    initial_speed
    * np.cos(initial_heading)
)

east = gps_east[first_index]

north = gps_north[first_index]

heading = initial_heading


# ============================================================
# 8. NHC PARAMETERS
# ============================================================

# NHC assumes that lateral velocity
# in the vehicle body frame is approximately zero.

NHC_GAIN = 0.85


# ============================================================
# 9. STORAGE
# ============================================================

estimated_east = []

estimated_north = []

estimated_time = []

position_errors = []


# ============================================================
# 10. LOOP
# ============================================================

previous_time = time[first_index]


for i in indices:

    current_time = time[i]

    dt = (
        current_time
        - previous_time
    )

    previous_time = current_time

    if dt <= 0:
        dt = 0.01


    # ========================================================
    # GNSS AVAILABLE
    # ========================================================

    gnss_available = not (
        blackout_start
        <= current_time
        <= blackout_end
    )


    if gnss_available:

        # GNSS provides absolute position.

        east = gps_east[i]

        north = gps_north[i]

        heading = np.deg2rad(
            gps_heading[i]
        )

        speed = (
            gps_speed[i]
            / 3.6
        )

        velocity_east = (
            speed
            * np.sin(heading)
        )

        velocity_north = (
            speed
            * np.cos(heading)
        )


    # ========================================================
    # GNSS BLACKOUT
    # ========================================================

    else:

        # ----------------------------------------------------
        # Update heading
        # ----------------------------------------------------

        heading += (
            gyro_yaw[i]
            * dt
        )


        # ----------------------------------------------------
        # Transform acceleration
        # ----------------------------------------------------

        acceleration_east = (
            linear_x[i]
            * np.sin(heading)
            +
            linear_y[i]
            * np.cos(heading)
        )

        acceleration_north = (
            linear_x[i]
            * np.cos(heading)
            -
            linear_y[i]
            * np.sin(heading)
        )


        # ----------------------------------------------------
        # Update velocity
        # ----------------------------------------------------

        velocity_east += (
            acceleration_east
            * dt
        )

        velocity_north += (
            acceleration_north
            * dt
        )


        # ====================================================
        # NHC
        # ====================================================

        # Convert navigation-frame velocity
        # into vehicle/body frame.

        forward_velocity = (
            velocity_east
            * np.sin(heading)
            +
            velocity_north
            * np.cos(heading)
        )

        lateral_velocity = (
            velocity_east
            * np.cos(heading)
            -
            velocity_north
            * np.sin(heading)
        )


        # NHC says:
        #
        # lateral velocity ≈ 0
        #
        # Therefore suppress the lateral component.

        corrected_lateral_velocity = (
            lateral_velocity
            * (1 - NHC_GAIN)
        )


        # Convert back to navigation frame.

        velocity_east = (
            forward_velocity
            * np.sin(heading)
            +
            corrected_lateral_velocity
            * np.cos(heading)
        )

        velocity_north = (
            forward_velocity
            * np.cos(heading)
            -
            corrected_lateral_velocity
            * np.sin(heading)
        )


        # ----------------------------------------------------
        # Update position
        # ----------------------------------------------------

        east += (
            velocity_east
            * dt
        )

        north += (
            velocity_north
            * dt
        )


    # ========================================================
    # STORE
    # ========================================================

    estimated_east.append(east)

    estimated_north.append(north)

    estimated_time.append(
        current_time
    )

    error = np.sqrt(
        (
            east
            - gps_east[i]
        ) ** 2
        +
        (
            north
            - gps_north[i]
        ) ** 2
    )

    position_errors.append(error)


# ============================================================
# 11. ARRAYS
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

position_errors = np.array(
    position_errors
)


# ============================================================
# 12. BLACKOUT ERROR
# ============================================================

blackout_mask = (
    (estimated_time >= blackout_start)
    &
    (estimated_time <= blackout_end)
)

blackout_errors = (
    position_errors[
        blackout_mask
    ]
)


print()
print("======================================")
print("NavDR — NHC FUSION")
print("======================================")

print(
    f"Blackout duration: "
    f"{blackout_duration} seconds"
)

print(
    f"Mean blackout error: "
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
# 13. TRAJECTORY
# ============================================================

plt.figure(figsize=(12, 8))

plt.plot(
    gps_east[indices],
    gps_north[indices],
    label="GNSS Reference"
)

plt.plot(
    estimated_east,
    estimated_north,
    label="NHC Navigation"
)

lost_index = np.where(
    estimated_time >= blackout_start
)[0][0]

restored_index = np.where(
    estimated_time > blackout_end
)[0][0]

plt.scatter(
    estimated_east[lost_index],
    estimated_north[lost_index],
    label="GNSS LOST"
)

plt.scatter(
    estimated_east[restored_index],
    estimated_north[restored_index],
    label="GNSS RESTORED"
)

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "NavDR — Non-Holonomic Constraint Navigation"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.show()


# ============================================================
# 14. ERROR
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    estimated_time,
    position_errors,
    label="NHC Position Error"
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
    "NavDR — NHC Position Error"
)

plt.legend()

plt.grid()

plt.show()