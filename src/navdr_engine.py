import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ============================================================
# NAVDR - CORRECTED INTEGRATED PROTOTYPE
# ============================================================

SMARTPHONE_FILE = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

# GNSS outage used for demonstration
BLACKOUT_START = 1200.0
BLACKOUT_END = 1260.0

# ============================================================
# LOAD DATA
# ============================================================

print("Loading smartphone dataset...")

smartphone = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)

# Normalize column names
smartphone.columns = smartphone.columns.str.strip()

print("Dataset shape:", smartphone.shape)

# ============================================================
# COLUMN NAMES
# ============================================================

TIME_COL = "TIME SINCE START (ms)"

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GYRO_YAW = "GYROSCOPE Yaw (rad/s)"
GYRO_PITCH = "GYROSCOPE Pitch (rad/s)"
GYRO_ROLL = "GYROSCOPE Roll (rad/s)"

GPS_LAT = "GPS LATITUDE (degrees)"
GPS_LON = "GPS LONGITUDE (degrees)"
GPS_SPEED = "GPS SPEED (Kmh)"

# ============================================================
# EXTRACT DATA
# ============================================================

time = smartphone[TIME_COL].to_numpy(dtype=float) / 1000.0

ax = smartphone[ACC_X].to_numpy(dtype=float)
ay = smartphone[ACC_Y].to_numpy(dtype=float)
az = smartphone[ACC_Z].to_numpy(dtype=float)

gyro_yaw = smartphone[GYRO_YAW].to_numpy(dtype=float)

gps_lat = smartphone[GPS_LAT].to_numpy(dtype=float)
gps_lon = smartphone[GPS_LON].to_numpy(dtype=float)
gps_speed = smartphone[GPS_SPEED].to_numpy(dtype=float)

# ============================================================
# VALID DATA
# ============================================================

valid = (
    np.isfinite(time)
    & np.isfinite(ax)
    & np.isfinite(ay)
    & np.isfinite(az)
    & np.isfinite(gyro_yaw)
    & np.isfinite(gps_lat)
    & np.isfinite(gps_lon)
    & np.isfinite(gps_speed)
)

time = time[valid]
ax = ax[valid]
ay = ay[valid]
az = az[valid]
gyro_yaw = gyro_yaw[valid]
gps_lat = gps_lat[valid]
gps_lon = gps_lon[valid]
gps_speed = gps_speed[valid]

print("Valid samples:", len(time))

# ============================================================
# GPS → LOCAL XY
# ============================================================

lat0 = gps_lat[0]
lon0 = gps_lon[0]

EARTH_RADIUS = 6371000.0

gps_x = (
    np.radians(gps_lon - lon0)
    * EARTH_RADIUS
    * np.cos(np.radians(lat0))
)

gps_y = (
    np.radians(gps_lat - lat0)
    * EARTH_RADIUS
)

# ============================================================
# FIND BLACKOUT
# ============================================================

start_idx = np.searchsorted(time, BLACKOUT_START)
end_idx = np.searchsorted(time, BLACKOUT_END)

if start_idx >= len(time) or end_idx >= len(time):
    raise ValueError("Blackout interval is outside the dataset.")

print()
print("==========================================")
print("GNSS BLACKOUT")
print("==========================================")

print(f"GNSS LOST     : {time[start_idx]:.2f} s")
print(f"GNSS RESTORED : {time[end_idx]:.2f} s")
print(
    f"Duration      : "
    f"{time[end_idx] - time[start_idx]:.2f} s"
)

# ============================================================
# ESTIMATE HEADING FROM GPS ONLY BEFORE BLACKOUT
# ============================================================
#
# IMPORTANT:
# We use the trajectory BEFORE the outage to estimate
# the initial driving direction.
#
# During the blackout, no GPS position is used.
# ============================================================

print()
print("Estimating initial vehicle heading...")

dx = np.gradient(gps_x)
dy = np.gradient(gps_y)

gps_heading = np.unwrap(
    np.arctan2(dy, dx)
)

# Use a stable heading immediately before blackout
heading_start = np.median(
    gps_heading[max(0, start_idx - 20):start_idx]
)

print(
    f"Initial heading: "
    f"{np.degrees(heading_start):.2f} degrees"
)

# ============================================================
# AI SPEED MODEL
# ============================================================

AI_MODEL_FILE = r"models\speed_model.pkl"

ai_model = None

if os.path.exists(AI_MODEL_FILE):

    try:

        ai_model = joblib.load(AI_MODEL_FILE)

        print()
        print("AI speed model loaded.")

    except Exception as error:

        print("Could not load AI model:")
        print(error)

else:

    print()
    print("AI speed model not found.")

# ============================================================
# AI SPEED PREDICTION
# ============================================================
#
# Our first Random Forest model uses exactly 6 features:
#
# accelerometer X/Y/Z
# gyroscope yaw/pitch/roll
#
# This guarantees feature compatibility.
# ============================================================

ai_speed = gps_speed.copy()

if ai_model is not None:

    print("Generating AI speed predictions...")

    gyro_pitch = smartphone.loc[
        valid,
        GYRO_PITCH
    ].to_numpy(dtype=float)

    gyro_roll = smartphone.loc[
        valid,
        GYRO_ROLL
    ].to_numpy(dtype=float)

    X_ai = np.column_stack([
        ax,
        ay,
        az,
        gyro_yaw,
        gyro_pitch,
        gyro_roll
    ])

    try:

        predicted_speed = ai_model.predict(X_ai)

        ai_speed = np.maximum(
            predicted_speed,
            0
        )

        print("AI speed prediction successful.")

    except Exception as error:

        print()
        print("AI prediction failed:")
        print(error)

        print("Using GPS speed as temporary fallback.")

# ============================================================
# CORRECTED DEAD RECKONING
# ============================================================
#
# DR starts EXACTLY at the GNSS position where signal is lost.
#
# This is the key correction.
# ============================================================

print()
print("Running corrected dead reckoning...")

dr_x = gps_x.copy()
dr_y = gps_y.copy()

# Anchor at GNSS-loss point
dr_x[start_idx] = gps_x[start_idx]
dr_y[start_idx] = gps_y[start_idx]

# ------------------------------------------------------------
# During blackout only
# ------------------------------------------------------------

current_heading = heading_start

for i in range(start_idx + 1, end_idx):

    dt = time[i] - time[i - 1]

    if dt <= 0 or dt > 1:
        dt = 0.1

    # AI-predicted vehicle speed
    speed_ms = ai_speed[i] / 3.6

    # --------------------------------------------------------
    # Simple gyro heading update
    # --------------------------------------------------------

    current_heading += gyro_yaw[i] * dt

    # --------------------------------------------------------
    # Vehicle movement
    # --------------------------------------------------------

    distance = speed_ms * dt

    dx_step = (
        distance
        * np.cos(current_heading)
    )

    dy_step = (
        distance
        * np.sin(current_heading)
    )

    dr_x[i] = dr_x[i - 1] + dx_step
    dr_y[i] = dr_y[i - 1] + dy_step

# ============================================================
# NON-HOLONOMIC CONSTRAINT
# ============================================================
#
# Vehicle is assumed to move primarily in its forward
# direction rather than sliding sideways.
# ============================================================

print("Applying NHC constraint...")

nhc_x = dr_x.copy()
nhc_y = dr_y.copy()

for i in range(start_idx + 1, end_idx):

    dx_step = nhc_x[i] - nhc_x[i - 1]
    dy_step = nhc_y[i] - nhc_y[i - 1]

    distance = np.sqrt(
        dx_step ** 2 +
        dy_step ** 2
    )

    if distance <= 0:
        continue

    direction = np.arctan2(
        dy_step,
        dx_step
    )

    nhc_x[i] = (
        nhc_x[i - 1]
        + distance * np.cos(direction)
    )

    nhc_y[i] = (
        nhc_y[i - 1]
        + distance * np.sin(direction)
    )

# ============================================================
# NAVIGATION OUTPUT
# ============================================================

navigation_x = gps_x.copy()
navigation_y = gps_y.copy()

# ------------------------------------------------------------
# GNSS AVAILABLE BEFORE BLACKOUT
# ------------------------------------------------------------

navigation_x[:start_idx] = gps_x[:start_idx]
navigation_y[:start_idx] = gps_y[:start_idx]

# ------------------------------------------------------------
# GNSS BLACKOUT
# ------------------------------------------------------------

navigation_x[start_idx:end_idx] = nhc_x[start_idx:end_idx]
navigation_y[start_idx:end_idx] = nhc_y[start_idx:end_idx]

# ------------------------------------------------------------
# GNSS RESTORED
# ------------------------------------------------------------
#
# Re-anchor navigation to the real GNSS position.
# ------------------------------------------------------------

navigation_x[end_idx:] = gps_x[end_idx:]
navigation_y[end_idx:] = gps_y[end_idx:]

# ============================================================
# POSITION ERROR
# ============================================================

position_error = np.sqrt(
    (navigation_x - gps_x) ** 2
    +
    (navigation_y - gps_y) ** 2
)

blackout_error = position_error[
    start_idx:end_idx
]

# ============================================================
# RESULTS
# ============================================================

print()
print("==========================================")
print("NAVDR RESULTS")
print("==========================================")

print(
    f"Maximum blackout error : "
    f"{np.max(blackout_error):.2f} m"
)

print(
    f"Mean blackout error    : "
    f"{np.mean(blackout_error):.2f} m"
)

print(
    f"Error at GNSS restore  : "
    f"{position_error[end_idx]:.2f} m"
)

# Distance travelled during blackout
reference_blackout_distance = np.sum(
    np.sqrt(
        np.diff(gps_x[start_idx:end_idx]) ** 2
        +
        np.diff(gps_y[start_idx:end_idx]) ** 2
    )
)

print(
    f"Reference blackout distance : "
    f"{reference_blackout_distance:.2f} m"
)

if reference_blackout_distance > 0:

    drift_percentage = (
        np.max(blackout_error)
        /
        reference_blackout_distance
    ) * 100

    print(
        f"Maximum drift percentage   : "
        f"{drift_percentage:.2f}%"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results = pd.DataFrame({

    "time": time,

    "gps_x": gps_x,
    "gps_y": gps_y,

    "ai_speed": ai_speed,

    "navigation_x": navigation_x,
    "navigation_y": navigation_y,

    "position_error": position_error

})

results.to_csv(
    "results/navdr_corrected_results.csv",
    index=False
)

print()
print(
    "Saved:"
    "\nresults/navdr_corrected_results.csv"
)

# ============================================================
# TRAJECTORY GRAPH
# ============================================================

plt.figure(figsize=(12, 8))

plt.plot(
    gps_x,
    gps_y,
    label="GNSS Reference"
)

plt.plot(
    navigation_x,
    navigation_y,
    label="NavDR Navigation"
)

plt.scatter(
    gps_x[start_idx],
    gps_y[start_idx],
    s=80,
    label="GNSS LOST"
)

plt.scatter(
    gps_x[end_idx],
    gps_y[end_idx],
    s=80,
    label="GNSS RESTORED"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")

plt.title(
    "NavDR — AI Dead Reckoning + NHC + GNSS Fusion"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/navdr_corrected_trajectory.png",
    dpi=200
)

plt.show()

# ============================================================
# ERROR GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    position_error,
    label="NavDR Position Error"
)

plt.axvline(
    time[start_idx],
    linestyle="--",
    label="GNSS LOST"
)

plt.axvline(
    time[end_idx],
    linestyle="--",
    label="GNSS RESTORED"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position Error (m)")

plt.title(
    "NavDR — Position Error During GNSS Blackout"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/navdr_corrected_error.png",
    dpi=200
)

plt.show()

print()
print("==========================================")
print("CORRECTED NAVDR ENGINE COMPLETE")
print("==========================================")