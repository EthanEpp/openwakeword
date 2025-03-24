import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Paste your data here
log_data = """
2025_03_14_20_50_55, 0.015712
2025_03_14_21_07_15, 0.019352
2025_03_14_22_37_36, 0.179349
2025_03_15_05_48_03, 0.249589
2025_03_15_08_24_09, 0.029735
2025_03_15_08_57_37, 0.995103
2025_03_15_10_54_32, 0.014319
2025_03_15_11_56_47, 0.719795
2025_03_15_13_46_58, 0.019648
2025_03_15_13_56_15, 0.877090

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
