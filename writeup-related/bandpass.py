import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def bandpass_filter(signal, fps, low=0.7, high=2.5):
    """
    Applies a bandpass filter to the given signal with given fps.
    """
    nyquist = 0.5 * fps
    b, a = scipy.signal.butter(3, [low/nyquist, high/nyquist], btype='bandpass')
    return scipy.signal.filtfilt(b, a, signal, axis=-1)


# 1. Setup Simulation Parameters
fps = 50.0                # Sampling rate (Hz)
duration = 10.0           # Total duration in seconds
time = np.arange(0, duration, 1/fps)

# 2. Synthesize Signals (Target frequency inside passband, others outside)
target_signal = np.sin(2 * np.pi * 1.5 * time)     # 1.5 Hz (INSIDE 0.7 - 2.5 Hz)
low_freq_noise = np.sin(2 * np.pi * 0.2 * time)    # 0.2 Hz (OUTSIDE low cutoff)
high_freq_noise = np.sin(2 * np.pi * 8.0 * time)   # 8.0 Hz (OUTSIDE high cutoff)
random_noise = np.random.normal(0, 0.4, len(time)) # High-frequency white noise

# Combine components into a noisy input signal
noisy_signal = target_signal + low_freq_noise + high_freq_noise + random_noise

# 3. Apply the Bandpass Filter
filtered_signal = bandpass_filter(noisy_signal, fps, low=0.7, high=2.5)

# 4. Visualize the Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Plot: Complex Noisy Input
ax1.plot(time, noisy_signal, label="Noisy Signal (Combined)", color="darkgray", alpha=0.7)
ax1.plot(time, target_signal, label="Pure Target Signal (1.5 Hz)", color="crimson", lw=2, linestyle="--")
ax1.set_title("Original Noisy Input Signal", fontsize=14, fontweight="bold")
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Bottom Plot: Clean Filtered Output
ax2.plot(time, filtered_signal, label="Filtered Signal (0.7 - 2.5 Hz)", color="dodgerblue", lw=2.5)
ax2.plot(time, target_signal, label="Pure Target Signal (Reference)", color="crimson", lw=1.5, linestyle="--")
ax2.set_title("Output After Bandpass Filtering", fontsize=14, fontweight="bold")
ax2.set_xlabel("Time (seconds)", fontsize=12)
ax2.set_ylabel("Amplitude", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()