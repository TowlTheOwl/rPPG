import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import scipy.signal
from utils.methods import bandpass_filter

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def rFFT(signal:np.ndarray, fps, min_hz=0.7, max_hz = 2.5, show_graph=False):
    """
    Extracts heart rate from a given 1d signal using rFFT

    1. apply Hanning window
    2. apply real Fast Fourier Transform (rFFT)
    3. find the frequency that maximizes the magnitude of the FFT
    4. return 60 * the frequency (the bpm)

    Input:
        signal: 1d signal that has been bandpassed, normalized, and detrended
    """

    # N = _next_power_of_2(len(signal))
    N = 4 * len(signal)
    windowed_signal = signal * np.hanning(len(signal))

    if show_graph:
        plt.plot(windowed_signal)
        plt.title("Windowed Signal")
        plt.show()

    fft_magnitudes = np.abs(rfft(windowed_signal, n=N))
    fft_frequencies = rfftfreq(N, d=1/fps)

    mask = (fft_frequencies >= min_hz) & (fft_frequencies <= max_hz)
    heart_rate_freqs = fft_frequencies[mask]
    heart_rate_mags = fft_magnitudes[mask]

    dominant_idx = np.argmax(heart_rate_mags)
    bpm = heart_rate_freqs[dominant_idx] * 60
    x, y = heart_rate_freqs[dominant_idx], heart_rate_mags[dominant_idx]

    if show_graph:
        plt.plot(heart_rate_freqs, heart_rate_mags)
        plt.title("Valid FFT Magnitudes")
        plt.annotate(f"Frequency Peak: {bpm}", xy=(x, y), xytext=(x+0.1, y-2), fontsize=5)
        plt.plot(x, y, marker='o', color='red', markersize=5)
        plt.show()

    return round(bpm, 2), (heart_rate_freqs, heart_rate_mags)

def periodogram(signal:np.ndarray, fps, min_hz=0.7, max_hz = 2.5, show_graph=False):
    """
    Uses SciPy's periodogram method to derive heart rate from BVP signal.
    """
    N = _next_power_of_2(len(signal))
    N = 4 * len(signal)
    freqs, psd = scipy.signal.periodogram(signal, fs=fps, nfft=N, detrend=False, window='hann')

    mask = (freqs >= min_hz) & (freqs <= max_hz)
    masked_freqs = freqs[mask]
    masked_psd = psd[mask]

    dominant_idx = np.argmax(masked_psd)
    bpm = masked_freqs[dominant_idx] * 60
    x, y = masked_freqs[dominant_idx], masked_psd[dominant_idx]

    if show_graph:
        plt.plot(masked_freqs, masked_psd)
        plt.title("Valid Periodogram PSD")
        plt.annotate(f"Frequency Peak: {bpm}", xy=(x, y), xytext=(x+0.1, y-2), fontsize=5)
        plt.plot(x, y, marker='o', color='red', markersize=5)
        plt.show()

    return round(bpm, 2), (masked_freqs, masked_psd)

def peak_detection(signal:np.ndarray, fps, min_hz=0.7, max_hz = 2.5, show_graph=False):
    """
    Uses SciPy's find_peaks method to manually locate peaks, use the average distance between peaks
    to calculate the heart rate.
    """
    signal = signal - np.mean(signal)
    if np.std(signal) != 0:
        signal = signal / np.std(signal)
    signal = bandpass_filter(signal, fps, min_hz, max_hz)
    initial_peaks, _ = scipy.signal.find_peaks(signal)
    peak_heights = signal[initial_peaks]
    avg_peak_height = np.mean(peak_heights)
    min_prominence = 0.25 * avg_peak_height
    peaks, _ = scipy.signal.find_peaks(signal, prominence=min_prominence) # , distance=fps/max_hz
    bpm = 60 / (np.mean(np.diff(peaks))/fps)
    return round(bpm, 2), (np.linspace(0, len(signal)/fps, len(signal)), signal)