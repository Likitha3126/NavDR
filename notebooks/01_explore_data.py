import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. File paths
# =========================

smartphone_path = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\S-Vta1a.csv"
)

vehicle_path = (
    r"data\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\Vta (Driver E)"
    r"\Vta01a"
    r"\V-Vta1a.csv"
)


# =========================
# 2. Load data
# =========================

smartphone = pd.read_csv(
    smartphone_path,
    encoding="cp1252"
)

vehicle = pd.read_csv(
    vehicle_path,
    encoding="cp1252"
)


# =========================
# 3. Remove accidental spaces
# =========================

smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()


# =========================
# 4. Display basic information
# =========================

print("SMARTPHONE")
print("Shape:", smartphone.shape)
print()

print(smartphone.columns.tolist())
print()

print("VEHICLE")
print("Shape:", vehicle.shape)
print()

print(vehicle.columns.tolist())


# =========================
# 5. Check missing values
# =========================

print("\nSMARTPHONE MISSING VALUES")
print(smartphone.isnull().sum())

print("\nVEHICLE MISSING VALUES")
print(vehicle.isnull().sum())


# =========================
# 6. Display first rows
# =========================

print("\nSMARTPHONE SAMPLE")
print(smartphone.head())

print("\nVEHICLE SAMPLE")
print(vehicle.head())