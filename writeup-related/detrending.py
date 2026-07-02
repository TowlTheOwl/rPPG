import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# 1. Generate synthetic time-series data with a strong upward trend
np.random.seed(42)
time = np.arange(0, 100)
noise = np.random.normal(0, 5, size=100)
seasonal_cycle = 15 * np.sin(2 * np.pi * time / 12)  # Cyclical pattern
underlying_trend = 0.8 * time                       # Linear trend

# Combine components to create the raw signal
raw_signal = underlying_trend + seasonal_cycle + noise

# 2. Perform Detrending using NumPy (Least-Squares Line Fitting)
# Fit a 1st-degree polynomial (a straight line: y = mx + c)
coefficients = np.polyfit(time, raw_signal, 1)
trend_line = np.polyval(coefficients, time)
detrended_signal = scipy.signal.detrend(raw_signal) # equivalent to detrended_signal = raw_signal - trend_line

# 3. Plot the results with Matplotlib
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Plot: Original Data and Estimated Trend
ax1.plot(time, raw_signal, label="Raw Signal (Trend + Cycle + Noise)", color="royalblue", lw=2)
ax1.plot(time, trend_line, label="Estimated Linear Trend", color="crimson", linestyle="--", lw=2.5)
ax1.set_title("Original Time Series with Trend Line", fontsize=14, fontweight="bold")
ax1.set_ylabel("Value", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# Bottom Plot: Detrended Data
ax2.plot(time, detrended_signal, label="Detrended Signal (Fluctuations Only)", color="forestgreen", lw=2)
ax2.axhline(0, color="black", linestyle="-", alpha=0.5)  # Baseline reference
ax2.set_title("Detrended Time Series (Stationary Mean)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Time Steps", fontsize=12)
ax2.set_ylabel("Value (Centered)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
