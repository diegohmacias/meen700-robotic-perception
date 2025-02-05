import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths (update if necessary)
accel_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Accelerometer.csv'
gyro_file  = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Gyroscope.csv'

# Load the CSV files into DataFrames
accel_data = pd.read_csv(accel_file)
gyro_data  = pd.read_csv(gyro_file)

# For simplicity, assume both sensors are sampled at the same time.
# (If not, you may need to interpolate or align the data.)
time = accel_data['seconds_elapsed'].to_numpy()
dt = np.diff(time, prepend=time[0])  # Compute time differences for integration

# --- Step 1: Compute Accelerometer-Based Angle Estimates ---
# Compute roll and pitch from the accelerometer:
# Here, we assume:
#   roll  = arctan2(accel_y, accel_z)
#   pitch = arctan2(-accel_x, sqrt(accel_y^2 + accel_z^2))
accel_x = accel_data['x'].to_numpy()
accel_y = accel_data['y'].to_numpy()
accel_z = accel_data['z'].to_numpy()

roll_acc = np.arctan2(accel_y, accel_z)
pitch_acc = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))

# --- Step 2: Integrate Gyroscope Data ---
# Extract gyroscope readings (angular velocity in rad/s).
# Here we assume:
#   Gyro reading about the x-axis corresponds to roll rate.
#   Gyro reading about the y-axis corresponds to pitch rate.
gyro_x = gyro_data['x'].to_numpy()  # roll rate (rad/s)
gyro_y = gyro_data['y'].to_numpy()  # pitch rate (rad/s)

# --- Step 3: Implement the Complementary Filter ---
# Tuning parameter: α (typically close to 1; e.g., 0.98 gives more weight to the gyro)
alpha = 0.98

# Initialize filtered angle arrays with the first accelerometer measurements
roll_filtered = [roll_acc[0]]
pitch_filtered = [pitch_acc[0]]

# Loop through the data (starting from the second sample)
for i in range(1, len(time)):
    # Integrate the gyro data to get an angle estimate
    roll_gyro = roll_filtered[i-1] + gyro_x[i] * dt[i]
    pitch_gyro = pitch_filtered[i-1] + gyro_y[i] * dt[i]
    
    # Blend with the accelerometer estimates
    roll_f = alpha * roll_gyro + (1 - alpha) * roll_acc[i]
    pitch_f = alpha * pitch_gyro + (1 - alpha) * pitch_acc[i]
    
    roll_filtered.append(roll_f)
    pitch_filtered.append(pitch_f)

roll_filtered = np.array(roll_filtered)
pitch_filtered = np.array(pitch_filtered)

# --- Step 4: Plot the Results ---
plt.figure(figsize=(12, 10))

# Plot for Roll
plt.subplot(2, 1, 1)
plt.plot(time, roll_acc, label='Accelerometer Roll')
plt.plot(time, roll_filtered, label='Filtered Roll', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Roll (rad)')
plt.title('Roll Angle Estimation')
plt.legend()
plt.grid(True)

# Plot for Pitch
plt.subplot(2, 1, 2)
plt.plot(time, pitch_acc, label='Accelerometer Pitch')
plt.plot(time, pitch_filtered, label='Filtered Pitch', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (rad)')
plt.title('Pitch Angle Estimation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
