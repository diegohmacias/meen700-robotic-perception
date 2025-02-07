import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
def synchronize_timestamps(accelerometer, gyroscope, groundtruth, sampling_rate=100):
    """
    Synchronize timestamps across accelerometer, gyroscope, and ground truth data.

    Parameters:
        accelerometer (pd.DataFrame): Accelerometer data with 'seconds_elapsed'.
        gyroscope (pd.DataFrame): Gyroscope data with 'seconds_elapsed'.
        groundtruth (pd.DataFrame): Ground truth data with 'seconds_elapsed'.
        sampling_rate (int): Desired sampling rate in Hz (default: 100).

    Returns:
        tuple: Synchronized accelerometer, gyroscope, and ground truth DataFrames.
    """
    # Determine the common time range
    start_time = max(accelerometer['seconds_elapsed'].min(), gyroscope['seconds_elapsed'].min(), groundtruth['seconds_elapsed'].min())
    end_time = min(accelerometer['seconds_elapsed'].max(), gyroscope['seconds_elapsed'].max(), groundtruth['seconds_elapsed'].max())

    # Define a common time base
    common_time = np.arange(start_time, end_time, 1 / sampling_rate)

    # Interpolate each dataset
    accelerometer = accelerometer.set_index('seconds_elapsed').reindex(common_time).interpolate().reset_index()
    accelerometer.rename(columns={'index': 'seconds_elapsed'}, inplace=True)

    gyroscope = gyroscope.set_index('seconds_elapsed').reindex(common_time).interpolate().reset_index()
    gyroscope.rename(columns={'index': 'seconds_elapsed'}, inplace=True)

    groundtruth = groundtruth.set_index('seconds_elapsed').reindex(common_time).interpolate().reset_index()
    groundtruth.rename(columns={'index': 'seconds_elapsed'}, inplace=True)

    return accelerometer, gyroscope, groundtruth

def low_pass_filter(data, cutoff, fs, order=2):
    """
    Apply a low-pass filter to the data.

    Parameters:
        data (pd.Series): Data to filter.
        cutoff (float): Cutoff frequency for the filter.
        fs (float): Sampling rate of the data.
        order (int): Order of the Butterworth filter.

    Returns:
        pd.Series: Filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def complementary_filter(accelerometer, gyroscope, timestamps, alpha=0.98, initial_roll=0, initial_pitch=0):
    """
    Estimate roll and pitch using a complementary filter.

    Parameters:
        accelerometer (pd.DataFrame): Accelerometer data with columns ['x', 'y', 'z'].
        gyroscope (pd.DataFrame): Gyroscope data with columns ['x', 'y', 'z'].
        timestamps (np.array): Common timestamps in seconds.
        alpha (float): Complementary filter coefficient.
        initial_roll (float): Initial guess for roll angle (in radians).
        initial_pitch (float): Initial guess for pitch angle (in radians).

    Returns:
        pd.DataFrame: Estimated roll and pitch over time.
    """
    roll = [initial_roll]  # Initialize roll with initial guess
    pitch = [initial_pitch]  # Initialize pitch with initial guess

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]  # Time step

        # Accelerometer-based roll and pitch
        roll_acc = np.arctan2(accelerometer['y'][i], accelerometer['z'][i])
        pitch_acc = np.arctan2(-accelerometer['x'][i], 
                               np.sqrt(accelerometer['y'][i]**2 + accelerometer['z'][i]**2))

        # Gyroscope-based roll and pitch (integration)
        roll_gyro = roll[-1] + gyroscope['x'][i] * dt
        pitch_gyro = pitch[-1] + gyroscope['y'][i] * dt

        # Complementary filter
        roll.append(alpha * roll_gyro + (1 - alpha) * roll_acc)
        pitch.append(alpha * pitch_gyro + (1 - alpha) * pitch_acc)

    return pd.DataFrame({'timestamp': timestamps, 'roll': roll, 'pitch': pitch})

def process_imu_data(accelerometer, gyroscope, groundtruth, fs=100, cutoff=5, alpha=0.98, initial_roll=0, initial_pitch=0):
    """
    Process IMU data to estimate roll and pitch using a complementary filter and compare with ground truth.

    Parameters:
        accelerometer (pd.DataFrame): Accelerometer data with columns ['seconds_elapsed', 'x', 'y', 'z'].
        gyroscope (pd.DataFrame): Gyroscope data with columns ['seconds_elapsed', 'x', 'y', 'z'].
        groundtruth (pd.DataFrame): Ground truth data with columns ['seconds_elapsed', 'roll', 'pitch'].
        fs (int): Sampling frequency (default: 100 Hz).
        cutoff (float): Cutoff frequency for the low-pass filter (default: 5 Hz).
        alpha (float): Complementary filter coefficient (default: 0.98).
        initial_roll (float): Initial guess for roll angle (in radians).
        initial_pitch (float): Initial guess for pitch angle (in radians).

    Returns:
        pd.DataFrame: DataFrame containing timestamps, estimated roll, estimated pitch, ground truth roll, and ground truth pitch.
    """
    # Synchronize timestamps
    accelerometer, gyroscope, groundtruth = synchronize_timestamps(accelerometer, gyroscope, groundtruth)

    # Apply low-pass filter
    accelerometer['x'] = low_pass_filter(accelerometer['x'], cutoff, fs)
    accelerometer['y'] = low_pass_filter(accelerometer['y'], cutoff, fs)
    accelerometer['z'] = low_pass_filter(accelerometer['z'], cutoff, fs)
    gyroscope['x'] = low_pass_filter(gyroscope['x'], cutoff, fs)
    gyroscope['y'] = low_pass_filter(gyroscope['y'], cutoff, fs)
    gyroscope['z'] = low_pass_filter(gyroscope['z'], cutoff, fs)

    # Estimate roll and pitch using complementary filter
    timestamps = accelerometer['seconds_elapsed']
    complementary_result = complementary_filter(accelerometer, gyroscope, timestamps, alpha, initial_roll, initial_pitch)

    # Add ground truth for comparison
    complementary_result['roll_ground_truth'] = groundtruth['roll']
    complementary_result['pitch_ground_truth'] = groundtruth['pitch']

    return complementary_result

# NOTE: The data files are not provided in this notebook. Please use your own data files for testing. 
#       Also, the following data files are already calibrated from the SensorLogger app and are in proper units.
# File paths
accelfile = 'LoggerData\Roll Changes\RightSide\phone_on_back_rightside_60deg-2025-02-05_16-30-43\Accelerometer.csv'
gyrofile = 'LoggerData\Roll Changes\RightSide\phone_on_back_rightside_60deg-2025-02-05_16-30-43\Gyroscope.csv'
orientfile = 'LoggerData\Roll Changes\RightSide\phone_on_back_rightside_60deg-2025-02-05_16-30-43\Orientation.csv'

# Load the data
accelerometer = pd.read_csv(accelfile)
gyroscope = pd.read_csv(gyrofile)
groundtruth = pd.read_csv(orientfile)

# Process the data
result = process_imu_data(accelerometer, gyroscope, groundtruth, initial_roll=0.0, initial_pitch=0.0)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(result['timestamp'], result['roll'], label='Roll (Complementary Filter)', color='blue')
plt.plot(result['timestamp'], result['pitch'], label='Pitch (Complementary Filter)', color='orange')
plt.plot(result['timestamp'], result['roll_ground_truth'], label='Roll (Ground Truth)', color='cyan', linestyle='dashed')
plt.plot(result['timestamp'], result['pitch_ground_truth'], label='Pitch (Ground Truth)', color='red', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Roll and Pitch Estimation using Complementary Filter (roll_change_60deg)')
plt.legend()
plt.grid(True)
plt.show()
