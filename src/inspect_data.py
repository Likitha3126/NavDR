import pandas as pd

smartphone_path = r"data\Synchronised V abd S datasets\Synchronised V abd S datasets\Categorised IOVNB Dataset\Vta (Driver E)\Vta01a\S-Vta1a.csv"

vehicle_path = r"data\Synchronised V abd S datasets\Synchronised V abd S datasets\Categorised IOVNB Dataset\Vta (Driver E)\Vta01a\V-Vta1a.csv"

smartphone = pd.read_csv(smartphone_path, encoding="cp1252")
vehicle = pd.read_csv(vehicle_path, encoding="cp1252")
print("\n========== SMARTPHONE DATA ==========")
print("Shape:", smartphone.shape)
print("\nColumns:")
print(smartphone.columns.tolist())
print("\nFirst 5 rows:")
print(smartphone.head())

print("\n========== VEHICLE DATA ==========")
print("Shape:", vehicle.shape)
print("\nColumns:")
print(vehicle.columns.tolist())
print("\nFirst 5 rows:")
print(vehicle.head())