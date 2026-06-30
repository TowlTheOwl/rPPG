import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate
from utils.post_processing import periodogram


def gt_heartrate_peak(gt_PPG, gt_time, start, end, show_graph=False):
    mask = (gt_time >= start) & (gt_time <= end)

    PPG_signal = gt_PPG[mask]
    timestamps = gt_time[mask]

    # Normalize data (zero mean and unit variance)
    PPG_signal = PPG_signal - np.mean(PPG_signal)
    if np.std(PPG_signal) != 0:
        PPG_signal = PPG_signal / np.std(PPG_signal)
    else:
        print("Warning: Standard deviation of PPG_signal is zero, normalization skipped.")

    initial_peaks, _ = find_peaks(PPG_signal)
    peak_heights = PPG_signal[initial_peaks]
    avg_peak_height = np.mean(peak_heights)
    min_prominence = 0.25 * avg_peak_height
    peaks, _ = find_peaks(PPG_signal, prominence=min_prominence)

    # 2. Get the timestamps of these peaks
    peak_times = timestamps[peaks]

    # 3. Calculate the time differences between consecutive peaks (in seconds)
    pp_intervals = np.diff(peak_times)

    # 4. Calculate the average PP interval and convert to BPM
    mean_pp_interval = np.mean(pp_intervals)
    heart_rate = 60 / mean_pp_interval


    if show_graph:
        print(f"Calculated Heart Rate: {heart_rate:.2f} BPM")

        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, PPG_signal, label='PPG Signal', color='tab:blue', linewidth=2)
        plt.plot(timestamps[peaks], PPG_signal[peaks], 'x', label='Detected Peaks', color='red', markersize=10, markeredgewidth=2.5)
        # 3. Format and label the chart axes
        plt.title('PPG Signal with Detected Systolic Peaks', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    return heart_rate

def gt_heartrate_fft(gt_PPG, gt_time, start, end, show_graph=False):
    mask = (gt_time >= start) & (gt_time <= end)

    PPG_signal = gt_PPG[mask]
    timestamps = gt_time[mask]

    fps = 30

    uniform_times = np.linspace(timestamps[0], timestamps[-1], num=fps*int(end-start))

    interpolator = interpolate.interp1d(timestamps, PPG_signal, kind='linear')
    uniform_values = interpolator(uniform_times)

    return periodogram(uniform_values, fps)[0]
    

def get_avg_hr(gt_hr, gt_time, start, end):
    mask = (gt_time >= start) & (gt_time <= end)

    hr_segment = gt_hr[mask]
    return np.mean(hr_segment)