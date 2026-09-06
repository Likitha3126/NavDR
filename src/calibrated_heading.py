import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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
# LOAD DATA
# ============================================================

df = pd.read_csv(
    SMARTPHONE_FILE,
    encoding="cp1252"
)

df.columns = df.columns.str.strip()


time = (
    df["TIME SINCE START (ms)"]
    .to_numpy(dtype=float)
    / 1000.0
)

lat = df["GPS LATITUDE (degrees)"].to_numpy(dtype=float)
lon = df["GPS LONGITUDE (degrees)"].to_numpy(dtype=float)

gyro_z = df["GYROSCOPE Yaw (rad/s)"].to_numpy(dtype=float)


# ============================================================
# GPS HEADING
# ============================================================

lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

dlat = np.gradient(lat_rad)
dlon = np.gradient(lon_rad)

mean_lat = np.nanmean(lat_rad)

east = dlon * np.cos(mean_lat)
north = dlat

gps_heading = np.degrees(
    np.arctan2(east, north)
)

gps_heading = (
    gps_heading + 360
) % 360


# ============================================================
# UNWRAP GPS HEADING
# ============================================================

gps_heading_unwrapped = np.degrees(
    np.unwrap(
        np.radians(gps_heading)
    )
)


# ============================================================
# SMOOTH GPS HEADING
# ============================================================

gps_heading_smooth = (
    pd.Series(gps_heading_unwrapped)
    .rolling(
        window=31,
        center=True,
        min_periods=1
    )
    .mean()
    .to_numpy()
)


# ============================================================
# GPS HEADING RATE
# ============================================================

dt = np.gradient(time)

dt = np.where(
    dt <= 0,
    0.1,
    dt
)

gps_heading_rate = (
    np.gradient(gps_heading_smooth)
    / dt
)


# Remove extreme GPS heading-rate spikes
valid_rate = (
    np.isfinite(gps_heading_rate)
    & np.isfinite(gyro_z)
    & (np.abs(gps_heading_rate) < 20)
)


# ============================================================
# BLACKOUT INDICES
# ============================================================

start_idx = np.searchsorted(
    time,
    BLACKOUT_START
)

end_idx = np.searchsorted(
    time,
    BLACKOUT_END
)


# ============================================================
# CALIBRATION PERIOD
# ============================================================
#
# IMPORTANT:
# Use data BEFORE GNSS loss.
#
# We learn:
#
# GPS heading rate = scale * gyro + offset
#
# ============================================================

calibration_mask = (
    valid_rate
    & (time >= BLACKOUT_START - 300)
    & (time < BLACKOUT_START - 10)
)


X = gyro_z[calibration_mask].reshape(-1, 1)

y = gps_heading_rate[calibration_mask]


print("========================================")
print("GYROSCOPE HEADING CALIBRATION")
print("========================================")

print(
    f"Calibration samples: {len(X)}"
)


# ============================================================
# FIT LINEAR CALIBRATION
# ============================================================

model = LinearRegression()

model.fit(X, y)


scale = model.coef_[0]
offset = model.intercept_


print(
    f"Learned scale  : {scale:.6f}"
)

print(
    f"Learned offset : {offset:.6f} deg/s"
)


# ============================================================
# CALIBRATED GYRO RATE
# ============================================================

calibrated_rate = (
    scale * gyro_z
    + offset
)


# ============================================================
# INITIAL HEADING
# ============================================================

initial_heading = (
    gps_heading_smooth[start_idx]
)


print(
    f"Initial GNSS heading: "
    f"{initial_heading:.2f} degrees"
)


# ============================================================
# INTEGRATE CALIBRATED HEADING
# ============================================================

heading = np.full(
    len(time),
    np.nan
)

heading[start_idx] = initial_heading


for i in range(
    start_idx + 1,
    end_idx
):

    delta_t = (
        time[i] - time[i - 1]
    )

    if delta_t <= 0 or delta_t > 1:
        delta_t = 0.1

    heading[i] = (
        heading[i - 1]
        + calibrated_rate[i] * delta_t
    )


# ============================================================
# COMPARE HEADING
# ============================================================

blackout_slice = slice(
    start_idx,
    end_idx
)


plt.figure(figsize=(12, 6))


plt.plot(
    time[blackout_slice],
    gps_heading_smooth[blackout_slice],
    label="GNSS-derived heading"
)


plt.plot(
    time[blackout_slice],
    heading[blackout_slice],
    label="Calibrated gyro heading"
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
plt.ylabel("Heading (degrees)")


plt.title(
    "NavDR — Calibrated Smartphone Heading"
)


plt.legend()
plt.grid(True)

plt.tight_layout()


plt.savefig(
    "results/calibrated_heading_v2.png",
    dpi=200
)


plt.show()


# ============================================================
# ERROR
# ============================================================

valid_compare = (
    np.isfinite(
        heading[blackout_slice]
    )
    &
    np.isfinite(
        gps_heading_smooth[blackout_slice]
    )
)


heading_error = (
    heading[blackout_slice][valid_compare]
    -
    gps_heading_smooth[blackout_slice][valid_compare]
)


print()
print("========================================")
print("HEADING ERROR")
print("========================================")

print(
    f"Mean absolute error: "
    f"{np.mean(np.abs(heading_error)):.2f} degrees"
)

print(
    f"Maximum absolute error: "
    f"{np.max(np.abs(heading_error)):.2f} degrees"
)

print(
    f"Final heading error: "
    f"{heading_error[-1]:.2f} degrees"
)