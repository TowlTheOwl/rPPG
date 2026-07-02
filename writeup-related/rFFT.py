import numpy as np
import matplotlib.pyplot as plt
# Import the rfft and rfftfreq functions explicitly from SciPy
from scipy.fft import rfft, rfftfreq

# 1. Setup Simulation Parameters
fps = 200.0               # Sampling rate (Hz)
duration = 10.0            # Total duration in seconds
time = np.arange(0, duration, 1/fps)
N = len(time)             # Total number of data samples

# 2. Create a Signal with Distinct Frequencies
# Composed of a 10 Hz wave, a 35 Hz wave, and random noise
signal_10hz = 2.5 * np.sin(2 * np.pi * 10 * time)
signal_35hz = 1.2 * np.sin(2 * np.pi * 35 * time)
noise = np.random.normal(0, 0.8, size=N)

clean_signal = signal_10hz + signal_35hz
noisy_signal = clean_signal + noise

# 3. Compute the Real Fast Fourier Transform (rFFT) using SciPy
# rfft processes real data and returns only the positive frequency terms
rfft_coefficients = rfft(noisy_signal)
frequencies = rfftfreq(N, d=1/fps)

# Normalize the magnitude scale to match original signal amplitudes
# Multiply by 2 because energy is split between positive and negative frequencies
magnitude = (2.0 / N) * np.abs(rfft_coefficients)

# 4. Plot Time Domain vs Frequency Domain
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top Plot: Time Domain Signal
ax1.plot(time, noisy_signal, label="Noisy Signal", color="gray", alpha=0.6)
ax1.plot(time, clean_signal, label="Clean Blueprint (10Hz + 35Hz)", color="darkorchid", lw=2)
ax1.set_title("Time Domain: Signal Over Time", fontsize=14, fontweight="bold")
ax1.set_xlabel("Time (seconds)", fontsize=12)
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Bottom Plot: Frequency Domain Spectrum
# Create the stem plot with default styles
markers, stems, base = ax2.stem(frequencies, magnitude)

# Apply custom 'teal' and 'black' styling safely
plt.setp(stems, color="teal", linestyle="-", linewidth=1.5)
plt.setp(markers, color="teal", marker="o", markersize=5)
plt.setp(base, color="black", linestyle="-", linewidth=1)

ax2.set_title("Frequency Domain: SciPy rFFT Magnitude Spectrum", fontsize=14, fontweight="bold")
ax2.set_xlabel("Frequency (Hz)", fontsize=12)
ax2.set_ylabel("Magnitude (Amplitude)", fontsize=12)
ax2.set_xlim(0, 60)  # Zoom into relevant frequency region (Nyquist limit is 100Hz)
ax2.grid(True, linestyle=":", alpha=0.6)

# Annotate peaks
ax2.annotate("10 Hz Peak", xy=(10, 2.5), xytext=(15, 2.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
ax2.annotate("35 Hz Peak", xy=(35, 1.2), xytext=(40, 1.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

plt.tight_layout()
plt.show()
