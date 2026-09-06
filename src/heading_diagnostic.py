import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMARTPHONE_FILE = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

BLACKOUT_START = 1200.0
BLACKOUT_END = 1260.0

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)

df.columns = df.columns.str.strip()

# ============================================================
# DATA
# ============================================================

time = (
    df["TIME SINCE START (ms)"]
    .to_numpy(dtype=float)
    / 1000.0
)

gps_speed = df[
    "GPS SPEED (Kmh)"
].to_numpy(dtype=float)

phone_yaw = df[
    "ORIENTATION (Yaw) (Â°)"
].to_numpy(dtype=float)

gyro_yaw = df[
    "GYROSCOPE Yaw (rad/s)"
].to_numpy(dtype=float)

# ============================================================
# FIND BLACKOUT
# ============================================================

start_idx = np.searchsorted(
    time,
    BLACKOUT_START
)

end_idx = np.searchsorted(
    time,
    BLACKOUT_END
)

print("====================================")
print("HEADING DIAGNOSTIC")
print("====================================")

print(
    "Blackout start:",
    time[start_idx]
)

print(
    "Blackout end:",
    time[end_idx]
)

# ============================================================
# PRINT VALUES
# ============================================================

print("\nPhone orientation yaw:")
print(
    phone_yaw[
        start_idx-10:start_idx+10
    ]
)

print("\nGyroscope yaw:")
print(
    gyro_yaw[
        start_idx-10:start_idx+10
    ]
)

print("\nGPS speed:")
print(
    gps_speed[
        start_idx-10:start_idx+10
    ]
)

# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n====================================")
print("STATISTICS")
print("====================================")

print(
    "Phone yaw before blackout:",
    np.nanmedian(
        phone_yaw[
            max(0, start_idx-100):start_idx
        ]
    )
)

print(
    "Phone yaw during blackout:",
    np.nanmedian(
        phone_yaw[
            start_idx:end_idx
        ]
    )
)

print(
    "Gyro yaw median:",
    np.nanmedian(
        gyro_yaw[
            start_idx:end_idx
        ]
    )
)

print(
    "GPS speed before blackout:",
    np.nanmedian(
        gps_speed[
            max(0, start_idx-100):start_idx
        ]
    )
)

print(
    "GPS speed during blackout:",
    np.nanmedian(
        gps_speed[
            start_idx:end_idx
        ]
    )
)

# ============================================================
# PLOT PHONE YAW
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    phone_yaw,
    label="Smartphone Orientation Yaw"
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

plt.xlabel("Time (seconds)")
plt.ylabel("Yaw (degrees)")

plt.title(
    "NavDR — Smartphone Orientation Yaw"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/heading_orientation.png",
    dpi=200
)

plt.show()

# ============================================================
# PLOT GYRO
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    gyro_yaw,
    label="Gyroscope Yaw Rate"
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

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")

plt.title(
    "NavDR — Smartphone Gyroscope Yaw"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/heading_gyro.png",
    dpi=200
)

plt.show()