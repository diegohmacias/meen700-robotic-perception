import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the CSV Files ---
# Paths (update these if necessary)
accel_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Accelerometer.csv'
gyro_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Gyroscope.csv'
orient_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Orientation.csv'

# Load data
accel_data = pd.read_csv(accel_file)
gyro_data = pd.read_csv(gyro_file)
orientation_data = pd.read_csv(orient_file)

# Extract timestamps
accel_time = accel_data['seconds_elapsed'].to_numpy()
gyro_time = gyro_data['seconds_elapsed'].to_numpy()
orientation_time = orientation_data['seconds_elapsed'].to_numpy()

# Extract calibrated accelerometer and gyroscope data
calibrated_accel_x = accel_data['x'].to_numpy()
calibrated_accel_y = accel_data['y'].to_numpy()
calibrated_accel_z = accel_data['z'].to_numpy()
calibrated_gyro_x = gyro_data['x'].to_numpy()
calibrated_gyro_y = gyro_data['y'].to_numpy()

# --- Step 2: Compute Accelerometer-Based Angles ---
# Compute roll and pitch angles from the accelerometer
roll_acc = np.arctan2(calibrated_accel_y, calibrated_accel_z)
pitch_acc = np.arctan2(-calibrated_accel_x, np.sqrt(calibrated_accel_y**2 + calibrated_accel_z**2))

# --- Step 3: Complementary Filtering ---
# Time differences for gyroscope integration
dt = np.diff(gyro_time, prepend=gyro_time[0])

# Initialize complementary filter
alpha = 0.98  # Tuning parameter
roll_filtered = [roll_acc[0]]
pitch_filtered = [pitch_acc[0]]

# Apply complementary filtering
for i in range(1, len(gyro_time)):
    # Integrate gyroscope data
    roll_gyro = roll_filtered[i - 1] + calibrated_gyro_x[i] * dt[i]
    pitch_gyro = pitch_filtered[i - 1] + calibrated_gyro_y[i] * dt[i]
    
    # Complementary filter
    roll_f = alpha * roll_gyro + (1 - alpha) * roll_acc[i]
    pitch_f = alpha * pitch_gyro + (1 - alpha) * pitch_acc[i]
    
    roll_filtered.append(roll_f)
    pitch_filtered.append(pitch_f)

roll_filtered = np.array(roll_filtered)
pitch_filtered = np.array(pitch_filtered)

# --- Step 4: Compare with iOS Reported Angles ---
# Extract iOS reported angles
reported_roll = orientation_data['roll'].to_numpy()
reported_pitch = orientation_data['pitch'].to_numpy()

# Interpolate filtered roll and pitch to match orientation timestamps
roll_filtered_interp = np.interp(orientation_time, gyro_time, roll_filtered)
pitch_filtered_interp = np.interp(orientation_time, gyro_time, pitch_filtered)

# --- Step 5: Plot Results ---
plt.figure(figsize=(12, 10))

# Roll comparison
plt.subplot(2, 1, 1)
plt.plot(orientation_time, reported_roll, label='iOS Reported Roll', linestyle='-', linewidth=1.5)
plt.plot(orientation_time, roll_filtered_interp, label='Filtered Roll (Complementary)', linestyle='--', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Roll (rad)')
plt.title('Roll Angle Comparison')
plt.legend()
plt.grid(True)

# Pitch comparison
plt.subplot(2, 1, 2)
plt.plot(orientation_time, reported_pitch, label='iOS Reported Pitch', linestyle='-', linewidth=1.5)
plt.plot(orientation_time, pitch_filtered_interp, label='Filtered Pitch (Complementary)', linestyle='--', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Pitch (rad)')
plt.title('Pitch Angle Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
