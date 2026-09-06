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

# IMPORTANT:
# Remove leading/trailing spaces from all column names.
smartphone.columns = smartphone.columns.str.strip()


# ============================================================
# 2. LOAD REQUIRED SENSOR DATA
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
# 4. SIMULATE GNSS BLACKOUT
# ============================================================

blackout_start = 1200
blackout_duration = 60
blackout_end = blackout_start + blackout_duration

mask = (
    (time >= blackout_start)
    &
    (time <= blackout_end)
)

indices = np.where(mask)[0]

if len(indices) == 0:
    raise RuntimeError(
        "No samples found inside the GNSS blackout interval."
    )

start_index = indices[0]
end_index = indices[-1]


# ============================================================
# 5. INITIAL GNSS STATE
# ============================================================

start_lat = latitude[start_index]
start_lon = longitude[start_index]

initial_speed = (
    gps_speed[start_index] / 3.6
)

initial_heading = np.deg2rad(
    gps_heading[start_index]
)


print()
print("================================")
print("NAVDR — AI + NHC FUSION")
print("================================")

print(
    f"Blackout: "
    f"{blackout_start}s → {blackout_end}s"
)

print(
    f"Initial speed: "
    f"{gps_speed[start_index]:.2f} km/h"
)

print(
    f"Initial heading: "
    f"{gps_heading[start_index]:.2f}°"
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
# 8. ADAPTIVE AI-STYLE MOTION FILTER
# ============================================================

def filter_acceleration(ax, ay, az):

    magnitude = np.sqrt(
        ax ** 2
        + ay ** 2
        + az ** 2
    )

    # Limit extreme acceleration spikes.
    #
    # This represents a lightweight motion-event
    # filtering stage suitable for the prototype.

    if magnitude > 12:

        scale = 12 / magnitude

        ax *= scale
        ay *= scale
        az *= scale

    return ax, ay, az


# ============================================================
# 9. DEAD RECKONING + NHC
# ============================================================

for i in range(
    start_index,
    end_index + 1
):

    # --------------------------------------------------------
    # Time difference
    # --------------------------------------------------------

    if i == start_index:

        dt = 0.0

    else:

        dt = (
            time[i]
            - time[i - 1]
        )

    # Protect against invalid time intervals.

    if dt <= 0 or dt > 1.0:

        dt = 0.1


    # --------------------------------------------------------
    # UPDATE HEADING
    # --------------------------------------------------------

    current_heading += (
        gyro_yaw[i] * dt
    )


    # --------------------------------------------------------
    # FILTER ACCELERATION
    # --------------------------------------------------------

    ax, ay, az = filter_acceleration(
        linear_x[i],
        linear_y[i],
        linear_z[i]
    )


    # --------------------------------------------------------
    # TRANSFORM ACCELERATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # UPDATE VELOCITY
    # --------------------------------------------------------

    velocity_east += (
        acceleration_east * dt
    )

    velocity_north += (
        acceleration_north * dt
    )


    # --------------------------------------------------------
    # NON-HOLONOMIC CONSTRAINT
    # --------------------------------------------------------
    #
    # Ground vehicles primarily move in their forward
    # direction and have very limited lateral motion.
    #
    # Therefore we reduce unrealistic lateral velocity.
    #
    # This is a simplified prototype implementation.
    # --------------------------------------------------------

    forward_velocity = (

        velocity_east
        * np.sin(current_heading)

        +

        velocity_north
        * np.cos(current_heading)
    )

    velocity_east = (
        forward_velocity
        * np.sin(current_heading)
    )

    velocity_north = (
        forward_velocity
        * np.cos(current_heading)
    )


    # --------------------------------------------------------
    # UPDATE POSITION
    # --------------------------------------------------------

    east += (
        velocity_east * dt
    )

    north += (
        velocity_north * dt
    )


    east_positions.append(east)
    north_positions.append(north)


# Convert to numpy arrays.

east_positions = np.array(
    east_positions
)

north_positions = np.array(
    north_positions
)


# ============================================================
# 10. GPS → LOCAL METERS
# ============================================================

earth_radius = 6371000.0

gps_lat = latitude[
    start_index:
    end_index + 1
]

gps_lon = longitude[
    start_index:
    end_index + 1
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
# 11. POSITION ERROR
# ============================================================

position_error = np.sqrt(

    (
        east_positions
        - gps_east
    ) ** 2

    +

    (
        north_positions
        - gps_north
    ) ** 2
)


final_error = position_error[-1]


reference_distance = np.sqrt(

    gps_east[-1] ** 2

    +

    gps_north[-1] ** 2
)


if reference_distance > 0:

    drift_percentage = (

        final_error
        / reference_distance
    ) * 100

else:

    drift_percentage = 0.0


# ============================================================
# 12. RESULTS
# ============================================================

print()
print("================================")
print("FUSION RESULT")
print("================================")

print(
    f"Reference displacement: "
    f"{reference_distance:.2f} m"
)

print(
    f"AI + NHC DR error: "
    f"{final_error:.2f} m"
)

print(
    f"Drift percentage: "
    f"{drift_percentage:.2f}%"
)

print(
    f"Number of blackout samples: "
    f"{len(east_positions)}"
)


# ============================================================
# 13. NAVIGATION TRAJECTORY
# ============================================================

plt.figure(
    figsize=(11, 8)
)

plt.plot(
    gps_east,
    gps_north,
    label="GNSS Reference"
)

plt.plot(
    east_positions,
    north_positions,
    label="AI + NHC Navigation"
)

plt.scatter(
    0,
    0,
    s=80,
    label="GNSS LOST"
)

plt.scatter(
    east_positions[-1],
    north_positions[-1],
    s=80,
    label="GNSS RESTORED"
)

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "NavDR — AI Dead Reckoning + NHC"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# 14. POSITION ERROR GRAPH
# ============================================================

blackout_time = time[
    start_index:
    end_index + 1
]


plt.figure(
    figsize=(11, 6)
)

plt.plot(
    blackout_time,
    position_error,
    label="NavDR Position Error"
)

plt.axvline(
    blackout_start,
    linestyle="--",
    label="GNSS LOST"
)

plt.axvline(
    blackout_end,
    linestyle="--",
    label="GNSS RESTORED"
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

plt.tight_layout()

plt.show()


print()
print("================================")
print("AI + NHC FUSION COMPLETE")
print("================================")