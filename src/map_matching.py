import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ============================================================
# LOAD DATA
# ============================================================

SMARTPHONE_PATH = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

smartphone = pd.read_csv(
    SMARTPHONE_PATH,
    encoding="cp1252"
)

smartphone.columns = smartphone.columns.str.strip()


# ============================================================
# GPS DATA
# ============================================================

lat = smartphone["GPS LATITUDE (degrees)"].to_numpy()
lon = smartphone["GPS LONGITUDE (degrees)"].to_numpy()

valid = (
    np.isfinite(lat)
    & np.isfinite(lon)
    & (lat != 0)
    & (lon != 0)
)

lat = lat[valid]
lon = lon[valid]


# ============================================================
# CONVERT LAT/LON TO LOCAL METERS
# ============================================================

lat0 = lat[0]
lon0 = lon[0]

earth_radius = 6371000

x = np.radians(lon - lon0) * earth_radius * np.cos(
    np.radians(lat0)
)

y = np.radians(lat - lat0) * earth_radius


# ============================================================
# CREATE SIMULATED DRIFT
# ============================================================

# This deliberately creates a drifting trajectory
# so that we can demonstrate map correction.

drift_x = np.linspace(0, 150, len(x))
drift_y = np.linspace(0, 100, len(y))

dr_x = x + drift_x
dr_y = y + drift_y


# ============================================================
# SIMPLE ROAD-LIKE MAP CONSTRAINT
# ============================================================

# For this first prototype we approximate the road network
# using the known trajectory itself.

# Sample reference points
road_points = np.column_stack((x, y))

tree = cKDTree(road_points)


# ============================================================
# MAP MATCHING
# ============================================================

dr_points = np.column_stack((dr_x, dr_y))

matched_points = []

MATCH_DISTANCE = 80.0

for point in dr_points:

    distance, index = tree.query(point)

    if distance < MATCH_DISTANCE:

        matched_points.append(
            road_points[index]
        )

    else:

        # If no nearby road is found,
        # retain the original DR position.

        matched_points.append(point)


matched_points = np.array(matched_points)

matched_x = matched_points[:, 0]
matched_y = matched_points[:, 1]


# ============================================================
# ERROR
# ============================================================

raw_error = np.sqrt(
    (dr_x - x) ** 2 +
    (dr_y - y) ** 2
)

matched_error = np.sqrt(
    (matched_x - x) ** 2 +
    (matched_y - y) ** 2
)


print("\n======================================")
print("MAP MATCHING RESULTS")
print("======================================")

print(
    f"Maximum DR error       : {raw_error.max():.2f} m"
)

print(
    f"Maximum matched error  : {matched_error.max():.2f} m"
)

print(
    f"Mean DR error          : {raw_error.mean():.2f} m"
)

print(
    f"Mean matched error     : {matched_error.mean():.2f} m"
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 8))

plt.plot(
    x,
    y,
    label="Reference Road / GPS"
)

plt.plot(
    dr_x,
    dr_y,
    label="Drifting Dead Reckoning"
)

plt.plot(
    matched_x,
    matched_y,
    label="Map-Matched Navigation"
)

plt.scatter(
    x[0],
    y[0],
    label="Start"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")

plt.title(
    "NavDR — Smart Map Matching"
)

plt.legend()
plt.grid(True)

plt.show()