import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

MODEL_PATH = r"models\speed_gru_v2.pth"
FEATURE_SCALER_PATH = r"models\speed_feature_scaler_v2.pkl"
TARGET_SCALER_PATH = r"models\speed_target_scaler_v2.pkl"

SEQ_LEN = 20

BLACKOUT_START = 1200.0
BLACKOUT_END = 1260.0

EARTH_RADIUS = 6378137.0


# ============================================================
# GRU
# ============================================================

class SpeedGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        out, _ = self.gru(x)

        last = out[:, -1, :]

        return self.output(last).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

print("\n==========================================")
print(" NavDR — AI Dead Reckoning Engine")
print("==========================================\n")

print("Loading dataset...")

df = pd.read_csv(
    CSV_PATH,
    encoding="cp1252"
)

df.columns = df.columns.str.strip()

print("Dataset loaded:", df.shape)


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

LAT_COL = "GPS LATITUDE (degrees)"
LON_COL = "GPS LONGITUDE (degrees)"

GPS_SPEED_COL = "GPS SPEED (Kmh)"


# ============================================================
# EXTRACT DATA
# ============================================================

time = (
    df[TIME_COL]
    .values
    .astype(float)
    / 1000.0
)

lat = df[LAT_COL].values.astype(float)

lon = df[LON_COL].values.astype(float)

gps_speed = (
    df[GPS_SPEED_COL]
    .values
    .astype(float)
)

acc = df[
    [
        ACC_X,
        ACC_Y,
        ACC_Z
    ]
].values.astype(float)

gyro = df[
    [
        GYRO_YAW,
        GYRO_PITCH,
        GYRO_ROLL
    ]
].values.astype(float)


# ============================================================
# CLEAN
# ============================================================

time = np.nan_to_num(time)

lat = np.nan_to_num(lat)

lon = np.nan_to_num(lon)

gps_speed = np.nan_to_num(gps_speed)

acc = np.nan_to_num(acc)

gyro = np.nan_to_num(gyro)


# ============================================================
# SAMPLING
# ============================================================

dt_array = np.diff(time)

valid_dt = dt_array[
    (dt_array > 0)
    &
    (dt_array < 1)
]

if len(valid_dt) > 0:

    dt_default = np.median(valid_dt)

else:

    dt_default = 0.1


print(
    f"Sampling interval: "
    f"{dt_default:.4f} s"
)

print(
    f"Sampling frequency: "
    f"{1.0 / dt_default:.2f} Hz"
)


# ============================================================
# LOAD AI MODEL
# ============================================================

print("\nLoading trained GRU model...")

device = torch.device("cpu")

model = SpeedGRU()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("GRU model loaded successfully.")


# ============================================================
# LOAD SCALERS
# ============================================================

print("Loading feature scaler...")

feature_scaler = joblib.load(
    FEATURE_SCALER_PATH
)

print("Feature scaler loaded.")

print("Loading target scaler...")

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

print("Target scaler loaded.")


# ============================================================
# AI FEATURES
# ============================================================

features = np.column_stack(
    [
        acc[:, 0],
        acc[:, 1],
        acc[:, 2],
        gyro[:, 0],
        gyro[:, 1],
        gyro[:, 2]
    ]
)


features_scaled = (
    feature_scaler.transform(features)
)


# ============================================================
# CREATE GRU SEQUENCES
# ============================================================

print("\nRunning AI speed prediction...")

X = []
indices = []

for i in range(
    SEQ_LEN - 1,
    len(df)
):

    X.append(
        features_scaled[
            i - SEQ_LEN + 1:
            i + 1
        ]
    )

    indices.append(i)


X = np.asarray(
    X,
    dtype=np.float32
)

print(
    "AI input shape:",
    X.shape
)


# ============================================================
# PREDICT
# ============================================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)


predictions = []


with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        512
    ):

        batch = X_tensor[
            start:start + 512
        ]

        output = model(batch)

        predictions.extend(
            output.numpy()
        )


predictions = np.asarray(
    predictions
)


# ============================================================
# INVERSE TARGET SCALING
# ============================================================

predictions_kmh = (
    target_scaler
    .inverse_transform(
        predictions.reshape(-1, 1)
    )
    .ravel()
)


ai_speed = np.zeros(
    len(df)
)

ai_speed[indices] = predictions_kmh


# Physical limits
ai_speed = np.clip(
    ai_speed,
    0,
    80
)


# Smooth AI output
ai_speed = (
    pd.Series(ai_speed)
    .rolling(
        7,
        min_periods=1
    )
    .mean()
    .values
)


# ============================================================
# BLACKOUT
# ============================================================

start_index = np.where(
    time >= BLACKOUT_START
)[0][0]

end_index = np.where(
    time >= BLACKOUT_END
)[0][0]


print("\n==========================================")
print(" GNSS BLACKOUT")
print("==========================================")

print(
    f"{BLACKOUT_START:.1f}s "
    f"→ "
    f"{BLACKOUT_END:.1f}s"
)


# ============================================================
# CONVERT GPS TO LOCAL METERS
# ============================================================

lat0 = lat[start_index]

lon0 = lon[start_index]


north_ref = (
    (lat - lat0)
    *
    np.pi
    /
    180.0
    *
    EARTH_RADIUS
)


east_ref = (
    (lon - lon0)
    *
    np.pi
    /
    180.0
    *
    EARTH_RADIUS
    *
    np.cos(
        np.radians(lat0)
    )
)


# ============================================================
# GPS COURSE / HEADING
#
# Calculate heading from consecutive GPS positions.
# ONLY USED BEFORE GNSS LOSS FOR CALIBRATION.
# ============================================================

gps_north_diff = np.diff(
    north_ref
)

gps_east_diff = np.diff(
    east_ref
)


gps_heading = np.zeros(
    len(df)
)


gps_heading[1:] = np.arctan2(
    gps_east_diff,
    gps_north_diff
)


# Remove noisy heading jumps
gps_heading_unwrapped = np.unwrap(
    gps_heading
)


# ============================================================
# HEADING RATE FROM GNSS
# ============================================================

gps_heading_rate = np.zeros(
    len(df)
)


for i in range(
    1,
    len(df)
):

    dt = (
        time[i]
        -
        time[i - 1]
    )

    if dt > 0:

        gps_heading_rate[i] = (
            gps_heading_unwrapped[i]
            -
            gps_heading_unwrapped[i - 1]
        ) / dt


# ============================================================
# PRE-BLACKOUT CALIBRATION WINDOW
# ============================================================

calibration_start_time = (
    BLACKOUT_START - 60.0
)

calibration_start = np.where(
    time >= calibration_start_time
)[0][0]


calibration_end = start_index


print("\nCalibrating gyro heading...")

print(
    f"Calibration window: "
    f"{calibration_start_time:.1f}s "
    f"→ "
    f"{BLACKOUT_START:.1f}s"
)


# ============================================================
# CALIBRATE GYRO SCALE + BIAS
#
# GPS heading rate = scale * gyro + offset
#
# Calibration happens ONLY before GNSS loss.
# ============================================================

gyro_cal = gyro[
    calibration_start:
    calibration_end
    + 1,
    0
]

gps_rate_cal = gps_heading_rate[
    calibration_start:
    calibration_end
    + 1
]


# Remove extreme GPS heading-rate outliers
valid = (
    np.isfinite(
        gyro_cal
    )
    &
    np.isfinite(
        gps_rate_cal
    )
    &
    (
        np.abs(
            gps_rate_cal
        )
        <
        2.0
    )
)


gyro_cal = gyro_cal[valid]

gps_rate_cal = gps_rate_cal[valid]


if len(gyro_cal) > 20:

    A = np.column_stack(
        [
            gyro_cal,
            np.ones(
                len(gyro_cal)
            )
        ]
    )

    scale, offset = np.linalg.lstsq(
        A,
        gps_rate_cal,
        rcond=None
    )[0]

else:

    scale = 1.0
    offset = 0.0


print(
    f"Gyro scale : {scale:.6f}"
)

print(
    f"Gyro offset: {offset:.6f}"
)


# ============================================================
# INITIAL HEADING
#
# Average GPS direction over last 5 seconds.
# ============================================================

initial_window = np.where(
    time >= BLACKOUT_START - 5.0
)[0]


initial_window = initial_window[
    initial_window < start_index
]


initial_heading = np.angle(
    np.mean(
        np.exp(
            1j
            *
            gps_heading[
                initial_window
            ]
        )
    )
)


print(
    f"Initial GNSS heading: "
    f"{np.degrees(initial_heading):.2f}°"
)


# ============================================================
# NAVIGATION ARRAYS
# ============================================================

nav_north = np.zeros(
    len(df)
)

nav_east = np.zeros(
    len(df)
)

nav_heading = np.zeros(
    len(df)
)


nav_heading[
    start_index
] = initial_heading


# ============================================================
# DEAD RECKONING
# ============================================================

print(
    "\nRunning AI dead reckoning..."
)


for i in range(
    start_index + 1,
    end_index + 1
):

    dt = (
        time[i]
        -
        time[i - 1]
    )


    if (
        dt <= 0
        or
        dt > 1
    ):

        dt = dt_default


    # --------------------------------------------------------
    # RAW GYRO
    # --------------------------------------------------------

    raw_gyro = gyro[
        i,
        0
    ]


    # --------------------------------------------------------
    # CALIBRATED GYRO RATE
    # --------------------------------------------------------

    calibrated_rate = (
        scale
        *
        raw_gyro
        +
        offset
    )


    # --------------------------------------------------------
    # UPDATE HEADING
    # --------------------------------------------------------

    nav_heading[i] = (
        nav_heading[i - 1]
        +
        calibrated_rate
        *
        dt
    )


    # --------------------------------------------------------
    # AI SPEED
    # --------------------------------------------------------

    speed_ms = (
        ai_speed[i]
        /
        3.6
    )


    # --------------------------------------------------------
    # NON-HOLONOMIC CONSTRAINT
    #
    # Vehicle is assumed to move mainly forward.
    # --------------------------------------------------------

    north_velocity = (
        speed_ms
        *
        np.cos(
            nav_heading[i]
        )
    )


    east_velocity = (
        speed_ms
        *
        np.sin(
            nav_heading[i]
        )
    )


    # --------------------------------------------------------
    # POSITION UPDATE
    # --------------------------------------------------------

    nav_north[i] = (
        nav_north[i - 1]
        +
        north_velocity
        *
        dt
    )


    nav_east[i] = (
        nav_east[i - 1]
        +
        east_velocity
        *
        dt
    )


# ============================================================
# BLACKOUT REFERENCE
# ============================================================

reference_north = north_ref[
    start_index:
    end_index + 1
]

reference_east = east_ref[
    start_index:
    end_index + 1
]


navigation_north = nav_north[
    start_index:
    end_index + 1
]

navigation_east = nav_east[
    start_index:
    end_index + 1
]


# ============================================================
# POSITION ERROR
# ============================================================

position_error = np.sqrt(
    (
        navigation_north
        -
        reference_north
    ) ** 2
    +
    (
        navigation_east
        -
        reference_east
    ) ** 2
)


mean_error = np.mean(
    position_error
)

max_error = np.max(
    position_error
)

final_error = position_error[-1]


# ============================================================
# REFERENCE DISTANCE
# ============================================================

reference_distance = np.sum(
    np.sqrt(
        np.diff(
            reference_north
        ) ** 2
        +
        np.diff(
            reference_east
        ) ** 2
    )
)


drift_percentage = (
    final_error
    /
    reference_distance
    *
    100
)


# ============================================================
# SPEED METRICS
# ============================================================

blackout_gps_speed = gps_speed[
    start_index:
    end_index + 1
]

blackout_ai_speed = ai_speed[
    start_index:
    end_index + 1
]


speed_mae = np.mean(
    np.abs(
        blackout_ai_speed
        -
        blackout_gps_speed
    )
)


speed_rmse = np.sqrt(
    np.mean(
        (
            blackout_ai_speed
            -
            blackout_gps_speed
        ) ** 2
    )
)


# ============================================================
# FINAL HEADING
# ============================================================

final_heading = (
    nav_heading[
        end_index
    ]
)


heading_change = (
    np.degrees(
        final_heading
        -
        initial_heading
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print(" FINAL NavDR RESULTS")
print("==========================================")

print(
    f"Blackout duration      : "
    f"{BLACKOUT_END - BLACKOUT_START:.1f} s"
)

print(
    f"Reference distance     : "
    f"{reference_distance:.2f} m"
)

print(
    f"AI speed MAE           : "
    f"{speed_mae:.2f} km/h"
)

print(
    f"AI speed RMSE          : "
    f"{speed_rmse:.2f} km/h"
)

print(
    f"Mean position error    : "
    f"{mean_error:.2f} m"
)

print(
    f"Maximum position error : "
    f"{max_error:.2f} m"
)

print(
    f"Final position error   : "
    f"{final_error:.2f} m"
)

print(
    f"Drift percentage       : "
    f"{drift_percentage:.2f}%"
)

print(
    f"Initial heading        : "
    f"{np.degrees(initial_heading):.2f}°"
)

print(
    f"Final heading          : "
    f"{np.degrees(final_heading):.2f}°"
)

print(
    f"Heading change         : "
    f"{heading_change:.2f}°"
)

print(
    "AI model               : GRU V2"
)

print(
    "NHC                    : Enabled"
)

print(
    "GNSS during blackout   : NOT USED"
)

print("==========================================\n")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results = pd.DataFrame(
    {
        "time":
            time[
                start_index:
                end_index + 1
            ],

        "reference_north":
            reference_north,

        "reference_east":
            reference_east,

        "nav_north":
            navigation_north,

        "nav_east":
            navigation_east,

        "position_error":
            position_error,

        "gps_speed":
            blackout_gps_speed,

        "ai_speed":
            blackout_ai_speed,

        "heading_deg":
            np.degrees(
                nav_heading[
                    start_index:
                    end_index + 1
                ]
            )
    }
)


results.to_csv(
    r"results\navdr_final_results.csv",
    index=False
)


print(
    r"Saved: results\navdr_final_results.csv"
)


# ============================================================
# PLOT 1 — TRAJECTORY
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.plot(
    reference_east,
    reference_north,
    label="GNSS Reference",
    linewidth=2
)

plt.plot(
    navigation_east,
    navigation_north,
    label="NavDR AI + NHC",
    linewidth=2
)

plt.scatter(
    0,
    0,
    s=100,
    label="GNSS LOST",
    zorder=5
)

plt.scatter(
    navigation_east[-1],
    navigation_north[-1],
    s=100,
    label="GNSS RESTORED",
    zorder=5
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

plt.grid(True)

plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2 — ERROR
# ============================================================

blackout_time = time[
    start_index:
    end_index + 1
]


plt.figure(
    figsize=(12, 7)
)

plt.plot(
    blackout_time,
    position_error,
    linewidth=2,
    label="NavDR Position Error"
)

plt.axvline(
    BLACKOUT_START,
    linestyle="--",
    label="GNSS LOST"
)

plt.axvline(
    BLACKOUT_END,
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

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3 — SPEED
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    blackout_time,
    blackout_gps_speed,
    linewidth=2,
    label="GNSS Speed Reference"
)

plt.plot(
    blackout_time,
    blackout_ai_speed,
    linewidth=2,
    label="AI Predicted Speed"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Speed (km/h)"
)

plt.title(
    "NavDR — AI Speed Estimation During GNSS Blackout"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 4 — HEADING
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    blackout_time,
    np.degrees(
        nav_heading[
            start_index:
            end_index + 1
        ]
    ),
    linewidth=2,
    label="NavDR Heading"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "NavDR — Estimated Heading During GNSS Blackout"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


print(
    "\nAll NavDR plots generated successfully."
)