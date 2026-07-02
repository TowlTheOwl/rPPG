import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Setup Simulation Parameters
fps = 100.0                # Sampling rate (Hz)
duration = 4.0             # Total duration in seconds
time = np.arange(0, duration, 1/fps)

# 2. Generate a 1D Cyclic Signal (Target frequency: 3.5 Hz)
target_frequency = 3.5
# Create a rhythmic wave with a bit of high-frequency noise
pure_signal = np.sin(2 * np.pi * target_frequency * time)
noise = np.random.normal(0, 0.15, size=len(time))
signal_1d = pure_signal + noise

# 3. Time-Domain Peak Detection
# height=0.3 filters out minor noise ripples near the baseline
# distance=15 ensures we don't double-count peaks within the same cycle
peak_indices, _ = find_peaks(signal_1d, height=0.3, distance=15)

# Convert peak indices into physical timestamps (seconds)
peak_times = time[peak_indices]

# 4. Calculate Frequency from Time Intervals
# Calculate the duration (Period, T) between each consecutive peak
periods = np.diff(peak_times)
average_period = np.mean(periods)

# Frequency is the reciprocal of the period (f = 1/T)
calculated_frequency = 1.0 / average_period

# Print the mathematical result
print("--- Time-Domain Frequency Analysis ---")
print(f"Actual Target Frequency:   {target_frequency} Hz")
print(f"Calculated Peak Frequency: {calculated_frequency:.3f} Hz")
print(f"Average Peak-to-Peak Time: {average_period:.3f} seconds")

# 5. Visualize the 1D Signal and Detected Peaks
fig, ax = plt.subplots(figsize=(11, 5))

# Plot the 1D continuous wave signal
ax.plot(time, signal_1d, label="1D Noisy Signal", color="darkslateblue", alpha=0.8, lw=1.5)

# Highlight the detected peak locations
ax.scatter(peak_times, signal_1d[peak_indices], color="crimson", marker="v", 
           s=120, zorder=3, label="Detected Peaks")

# Draw visual lines showing the Peak-to-Peak interval (Period)
if len(peak_times) > 1:
    ax.annotate('', xy=(peak_times[0], 1.2), xytext=(peak_times[1], 1.2),
                arrowprops=dict(arrowstyle='<->', color='forestgreen', lw=2))
    ax.text((peak_times[0] + peak_times[1])/2, 1.25, f"Period (T) ≈ {average_period:.3f}s", 
            color='forestgreen', fontweight='bold', ha='center')

# Style the chart
ax.set_title(f"Time-Domain Peak Detection (Estimated Freq: {calculated_frequency:.2f} Hz)", 
             fontsize=14, fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)
ax.set_ylim(-1.6, 1.6)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", fontsize=11)

plt.tight_layout()
plt.show()
