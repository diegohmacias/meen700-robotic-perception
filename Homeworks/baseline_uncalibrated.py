import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the CSV Files ---
# Paths (update these if necessary)
accel_uncalibrated_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/AccelerometerUncalibrated.csv'
gyro_uncalibrated_file  = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/GyroscopeUncalibrated.csv'
gravity_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Gravity.csv'
orient_file = 'LoggerData/Baseline/phone_on_back_0deg-2025-02-05_16-10-00/Orientation.csv'

accel_uncalibrated_data = pd.read_csv(accel_uncalibrated_file)
gravity_data = pd.read_csv(gravity_file)
gyro_uncalibrated_data = pd.read_csv(gyro_uncalibrated_file)
orientation_data = pd.read_csv(orient_file)

# --- Step 1: Calibrate Accelerometer ---
# Convert gravity data to g units for comparison
gravity_g = gravity_data[['x', 'y', 'z']].to_numpy() / 9.80665

# Uncalibrated accelerometer readings
accel_uncalibrated = accel_uncalibrated_data[['x', 'y', 'z']].to_numpy()

# Estimate accelerometer bias as the mean difference during static baseline
accel_bias = accel_uncalibrated.mean(axis=0) - gravity_g.mean(axis=0)

# Remove bias from uncalibrated accelerometer readings
accel_calibrated = accel_uncalibrated - accel_bias

# --- Step 2: Calibrate Gyroscope ---
# Uncalibrated gyroscope readings
gyro_uncalibrated = gyro_uncalibrated_data[['x', 'y', 'z']].to_numpy()

# Estimate gyroscope bias as the mean during static baseline
gyro_bias = gyro_uncalibrated.mean(axis=0)

# Remove bias from uncalibrated gyroscope readings
gyro_calibrated = gyro_uncalibrated - gyro_bias

# --- Step 3: Compute Total Acceleration ---
# Interpolate gravity data onto the accelerometer's timestamps
gravity_time = gravity_data['seconds_elapsed'].to_numpy()
accel_time = accel_uncalibrated_data['seconds_elapsed'].to_numpy()

gravity_interp = {
    'x': np.interp(accel_time, gravity_time, gravity_data['x'].to_numpy()),
    'y': np.interp(accel_time, gravity_time, gravity_data['y'].to_numpy()),
    'z': np.interp(accel_time, gravity_time, gravity_data['z'].to_numpy())
}

# Create a new gravity vector with the interpolated data
gravity_vector_interp = np.column_stack((gravity_interp['x'], gravity_interp['y'], gravity_interp['z']))

# Compute total acceleration
total_acceleration = gravity_vector_interp + accel_calibrated * 9.80665  # Convert calibrated accel back to m/s²

# --- Step 4: Complementary Filtering ---
# Extract calibrated accelerometer and gyroscope data
calibrated_accel_x = accel_calibrated[:, 0]
calibrated_accel_y = accel_calibrated[:, 1]
calibrated_accel_z = accel_calibrated[:, 2]
calibrated_gyro_x = gyro_calibrated[:, 0]
calibrated_gyro_y = gyro_calibrated[:, 1]

# Time differences for gyroscope integration
dt = np.diff(accel_time, prepend=accel_time[0])

# Compute accelerometer-based roll and pitch
roll_acc = np.arctan2(calibrated_accel_y, calibrated_accel_z)
pitch_acc = np.arctan2(-calibrated_accel_x, np.sqrt(calibrated_accel_y**2 + calibrated_accel_z**2))

# Initialize complementary filter
alpha = 0.98  # Tuning parameter
roll_filtered = [roll_acc[0]]
pitch_filtered = [pitch_acc[0]]

# Apply complementary filtering
for i in range(1, len(accel_time)):
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

# --- Step 5: Compare with iOS Reported Angles ---
# Extract iOS reported angles
reported_roll = orientation_data['roll'].to_numpy()
reported_pitch = orientation_data['pitch'].to_numpy()
orientation_time = orientation_data['seconds_elapsed'].to_numpy()

# Interpolate filtered roll and pitch to match orientation timestamps
roll_filtered_interp = np.interp(orientation_time, accel_time, roll_filtered)
pitch_filtered_interp = np.interp(orientation_time, accel_time, pitch_filtered)

# --- Step 6: Plot Results ---
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