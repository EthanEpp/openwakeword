import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Paste your data here
log_data = """
2025_03_03_22_28_05, 0.101367
2025_03_04_00_11_27, 0.023254
2025_03_04_01_11_04, 0.034653
2025_03_04_02_22_27, 0.166195
2025_03_04_02_26_38, 0.976987
2025_03_04_02_42_00, 0.014028
2025_03_04_03_31_08, 0.248988
2025_03_04_04_17_11, 0.018966
2025_03_04_04_43_50, 0.157840
2025_03_04_04_54_45, 0.412438
2025_03_04_06_21_53, 0.018451
2025_03_04_06_26_37, 0.299799
2025_03_04_06_52_31, 0.021767
2025_03_04_07_22_13, 0.042027
2025_03_04_07_38_27, 0.997885
2025_03_05_17_14_09, 0.917621
2025_03_05_22_18_22, 0.145357
2025_03_05_23_32_38, 0.100891
2025_03_06_00_29_03, 0.026051
2025_03_06_04_48_05, 0.025468
2025_03_06_05_44_40, 0.104732
2025_03_06_06_55_48, 0.028750
2025_03_04_18_38_49, 0.049567
2025_03_04_18_39_38, 0.031424
2025_03_04_20_21_50, 0.019523
2025_03_04_20_24_04, 0.998839
2025_03_04_21_51_50, 0.069593
2025_03_05_00_35_37, 0.013661
2025_03_05_03_18_22, 0.234742
2025_03_05_04_15_45, 0.016495
2025_03_05_04_29_19, 0.989551
2025_03_05_05_08_34, 0.019251

"""

# Convert log data to DataFrame
data = [line.split(", ") for line in log_data.strip().split("\n")]
df = pd.DataFrame(data, columns=["timestamp", "score"])

# Convert timestamp to datetime and score to float
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y_%m_%d_%H_%M_%S")
df["score"] = df["score"].astype(float)

# Sort by score in descending order (so thresholding removes lower scores first)
df = df.sort_values(by="score", ascending=False)

# Ask user for total hours of data
total_hours = int(input("Enter the total number of hours covered in the dataset: "))

# Generate thresholds from the unique scores
thresholds = np.sort(df["score"].unique())[::-1]  # Sorted descending

# Compute false accept rate per hour for each threshold
false_accept_rates = [df[df["score"] >= t].shape[0] / total_hours for t in thresholds]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(thresholds, false_accept_rates, marker="o", linestyle="-")

# Formatting
plt.xlabel("Threshold")
plt.ylabel("False Accepts per Hour")
plt.title("Cumulative False Accept Rate vs. Threshold")
plt.grid(True)

# Fix the x-axis formatting
plt.xscale("log")  # Keep log scale (or remove this line for linear scale)
plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())  # Ensures readable labels

plt.show()
