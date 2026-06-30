import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math

def bandpass_filter(signal, fps, low=0.7, high=2.5):
    """
    Applies a bandpass filter to the given signal with given fps.
    """
    nyquist = 0.5 * fps
    b, a = scipy.signal.butter(3, [low/nyquist, high/nyquist], btype='bandpass')
    return scipy.signal.filtfilt(b, a, signal, axis=-1)

def green_only(raw_green_signal:np.ndarray, fps:int, show_graph:bool=False):
    """
    Computes an rPPG signal using the Green-only method.

    This method exploits the fact that the green channel exhibits the
    strongest pulsatile component due to hemoglobin light absorption.

    Steps:
        1. Detrend the raw green signal to remove slow illumination drift.
        2. Restore the DC level and normalize by the mean to isolate
           relative intensity changes.
        3. Apply a bandpass filter to isolate heart-rate frequencies.
        4. Return the filtered signal as the blood volume pulse (BVP).

    Inputs:
        raw_green_signal (np.ndarray):
            Raw green-channel intensity values over time (shape: N,).
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, intermediate and final signals are plotted.

    Returns:
        np.ndarray:
            Bandpass-filtered green-channel rPPG signal.
    """
    
    G_d = scipy.signal.detrend(raw_green_signal) + np.mean(raw_green_signal)
    G_n = (G_d / np.mean(G_d)) - 1
    if show_graph:
        plt.plot(G_n)
        plt.title("GREEN: Normalized & Detrended Signal")
        plt.show()

    G_f = bandpass_filter(G_n, fps)
    if show_graph:
        plt.plot(G_f)
        plt.title("GREEN: Bandpassed Signal")
        plt.show()

    return G_f

def ratio_method(color_signal:np.ndarray, secondary_color:int, fps:int, show_graph=False):
    """
    Computes an rPPG signal using the color ratio method
    (Green/Red or Green/Blue).

    This method reduces motion and illumination artifacts by forming
    a ratio between normalized color channels.

    Steps:
        1. Normalize green and secondary color channels by their means.
        2. Compute the ratio signal:
            ratio = (G / mean(G)) / (C / mean(C)) - 1
        3. Detrend the ratio signal to remove low-frequency components.
        4. Apply a bandpass filter to isolate heart-rate frequencies.
        5. Return the filtered ratio signal.

    Inputs:
        color_signal (np.ndarray):
            RGB color signals with shape (3, N).
        secondary_color (int):
            Index of secondary channel:
                0 = red, 2 = blue.
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, the final rPPG signal is plotted.

    Returns:
        np.ndarray:
            Bandpass-filtered ratio-based rPPG signal.

    Raises:
        ValueError:
            If secondary_color is not 0 (red) or 2 (blue).
    """
    if not secondary_color in (0, 2):
        raise ValueError("seconary must be 0 or 2 (red or blue, respectively)")

    G = color_signal[1]
    C = color_signal[secondary_color]

    G_dc = np.mean(G)
    C_dc = np.mean(C)

    ratio_signal = (G / G_dc)/(C / C_dc) - 1

    S = bandpass_filter(scipy.signal.detrend(ratio_signal), fps)

    if show_graph:
        plt.plot(S)
        plt.title("Ratio: Bandpasssed Signal")
        plt.show()

    return S

def CHROM_method(color_signal:np.ndarray, fps:int, show_graph:bool=False):
    """
    Computes an rPPG signal using the CHROM (Chrominance-based) method.

    The CHROM method projects normalized RGB signals onto chrominance
    subspaces designed to suppress motion-induced intensity variations.

    Steps:
        1. Normalize each RGB channel by its temporal mean.
        2. Project normalized RGB onto chrominance signals:
            X_s = 3 R_n - 2 G_n
            Y_s = 1.5 R_n + G_n - 1.5 B_n
        3. Bandpass filter X_s and Y_s to obtain X_f and Y_f.
        4. Compute scaling factor:
            alpha = std(X_f) / std(Y_f)
        5. Form the pulse signal:
            S = X_f - alpha * Y_f
        6. Detrend & Bandpass the final signal
        7. Return the resulting rPPG signal.

    Inputs:
        color_signal (np.ndarray):
            RGB color signals with shape (3, N).
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, intermediate and final signals are plotted.

    Returns:
        np.ndarray:
            1D CHROM-based rPPG signal.
    """

    C_n = (color_signal / np.mean(color_signal, axis=1)[:, np.newaxis]) # (3, N)
    chrom_matrix = np.array([[3, -2, 0],[1.5, 1, -1.5]]) # (2, 3)
    XY_s = chrom_matrix @ C_n # (2, N)
    if show_graph:
        plt.plot(XY_s[0])
        plt.plot(XY_s[1])
        plt.title("CHROM: X and Y")
        plt.show()

    XY_f = bandpass_filter(XY_s, fps)
    if show_graph:
        plt.plot(XY_f[0])
        plt.plot(XY_f[1])
        plt.title("CHROM: X and Y bandpassed")
        plt.show()

    stdev = np.std(XY_f, axis=1)
    alpha = stdev[0]/stdev[1]
    S = XY_f[0] - alpha * XY_f[1]
    if show_graph:
        plt.plot(S)
        plt.title("CHROM: 1D signal")
        plt.show()

    S = scipy.signal.detrend(S)
    S = bandpass_filter(S, fps)  # re-filter after detrend
    return S

def POS_method(color_signal:np.ndarray, fps:int, show_graph:bool=False):
    """
    Computes an rPPG signal using the POS (Plane-Orthogonal-to-Skin) method.

    POS suppresses motion artifacts by projecting color variations onto
    a plane orthogonal to the skin-tone direction.

    Steps:
        1. Normalize each RGB channel by its temporal mean.
        2. Project normalized RGB onto orthogonal chrominance signals:
            X_s = G_n - B_n
            Y_s = -2 R_n + G_n + B_n
        3. Compute scaling factor:
            alpha = std(X_f) / std(Y_f)
        4. Form the pulse signal:
            S = X_f + alpha * Y_f
        5. Detrend and bandpass filter the result.
        6. Return the final rPPG signal.

    Inputs:
        color_signal (np.ndarray):
            RGB color signals with shape (3, N).
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, intermediate and final signals are plotted.

    Returns:
        np.ndarray:
            1D POS-based rPPG signal.
    """

    C_n = (color_signal / np.mean(color_signal, axis=1)[:, np.newaxis]) # (3, N)
    chrom_matrix = np.array([[0, 1, -1],[-2, 1, 1]]) # (2, 3)
    XY_s = chrom_matrix @ C_n # (2, N)
    if show_graph:
        plt.plot(XY_s[0])
        plt.plot(XY_s[1])
        plt.title("POS: X and Y")
        plt.show()
    # XY_f = bandpass_filter(XY_s, fps)
    XY_f = XY_s
    stdev = np.std(XY_f, axis=1)
    alpha = stdev[0]/stdev[1]
    S = XY_f[0] + alpha * XY_f[1]
    S = scipy.signal.detrend(S)
    S = bandpass_filter(S, fps)
    if show_graph:
        plt.plot(S)
        plt.title("POS: 1D signal")
        plt.show()

    return S

def CHROM_method_windowed(color_signal:np.ndarray, fps:int, window_len:float=1.6, show_graph:bool=False):
    """
    Computes heart rate using the windowed CHROM (Chrominance-based) method.

    This implementation applies the CHROM algorithm on overlapping temporal
    windows and reconstructs a continuous rPPG signal using overlap-add
    with a tapering window.

    Steps:
        1. Define window length in samples as window_len * fps
           and use 50% overlap between adjacent windows.
        2. For each temporal window:
            a. Extract windowed RGB signals.
            b. Normalize each channel by dividing by its temporal mean.
            c. Project normalized RGB onto two chrominance signals:
                X_s = 3 R_n - 2 G_n
                Y_s = 1.5 R_n + G_n - 1.5 B_n
            d. Bandpass filter X_s and Y_s.
            e. Compute alpha = std(X_f) / std(Y_f).
            f. Form windowed pulse signal:
                S_w = X_f - alpha * Y_f
            g. Remove window mean to avoid DC accumulation.
        3. Apply a Hanning tapering window and accumulate windowed signals
           using overlap-add reconstruction.
        4. Normalize by the accumulated window weights.
        5. Detrend and bandpass filter the reconstructed signal.
        6. Return the final 1D rPPG signal.

    Notes:
        - Windowing improves robustness to motion and illumination changes.
        - 50% overlap ensures smooth temporal reconstruction.
        - The Hanning window reduces boundary artifacts between windows.

    Inputs:
        color_signal (np.ndarray):
            RGB color signals with shape (3, N).
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, intermediate and final signals are plotted.

    Returns:
        np.ndarray:
            1D POS-based rPPG signal.
    """
    N = color_signal.shape[1]
    window_len = math.ceil(window_len * fps)
    step = window_len // 2
    S = np.zeros(N)
    W = np.zeros(N)
    taper_window = np.hanning(window_len) 

    chrom_matrix = np.array([[3, -2, 0],[1.5, 1, -1.5]]) # (2, 3)
    
    for i in range(0, N - window_len + 1, step):
        Cw = color_signal[:, i:i+window_len]
        Cn = Cw / np.mean(Cw, axis=1, keepdims=True)

        XY = bandpass_filter(chrom_matrix @ Cn, fps)
        stdev = np.std(XY, axis=1)
        alpha = stdev[0]/stdev[1]
        Sw = XY[0] - alpha * XY[1]
        Sw = Sw - np.mean(Sw)

        S[i:i+window_len] += Sw * taper_window
        W[i:i+window_len] += taper_window

    
    W = np.where(W == 0, 1e-6, W)
    S /= W
    valid_end = (((N - window_len) // step) * step) + window_len
    S = S[:valid_end]
    S = scipy.signal.detrend(S)
    S = bandpass_filter(S, fps)
    if show_graph:
        plt.plot(S)
        plt.title("CHROM Windowed: 1D signal")
        plt.show()

    return S

def POS_method_windowed(color_signal:np.ndarray, fps:int, window_len:float=1.6, show_graph:bool=False):
    """
    Computes heart rate using the windowed POS (Plane-Orthogonal-to-Skin) method.

    This implementation applies the POS algorithm on overlapping temporal
    windows and reconstructs a continuous rPPG signal via overlap-add.

    Steps:
        1. Define window length as 1.6 seconds (converted to samples)
           and use 1 frame step size.
        2. For each temporal window:
            a. Extract windowed RGB signals.
            b. Normalize each channel by dividing by its temporal mean.
            c. Project normalized RGB onto two orthogonal chrominance signals:
                X_s = G_n - B_n
                Y_s = -2 R_n + G_n + B_n
            d. Compute alpha = std(X_s) / std(Y_s).
            e. Form windowed pulse signal:
                S_w = X_s + alpha * Y_s
            f. Remove window mean to suppress DC components.
        3. Accumulate windowed signals using overlap-add reconstruction.
        4. Detrend and bandpass filter the reconstructed signal.
        5. Return the final 1D rPPG signal.

    Notes:
        - POS is designed to suppress motion-induced color variations by
          projecting onto a plane orthogonal to skin tone.
        - Windowing increases temporal adaptivity compared to the
          non-windowed POS formulation.
        - Unlike CHROM, no explicit tapering window is applied here.
    
    Inputs:
        color_signal (np.ndarray):
            RGB color signals with shape (3, N).
        fps (int):
            Frames per second of the video.
        show_graph (bool):
            If True, intermediate and final signals are plotted.

    Returns:
        np.ndarray:
            1D POS-based rPPG signal.
    """
    N = color_signal.shape[1]
    window_len = math.ceil(window_len * fps)
    # step = window_len // 2
    step = 1
    S = np.zeros(N)
    W = np.zeros(N)

    chrom_matrix = np.array([[0, 1, -1],[-2, 1, 1]]) # (2, 3)
    
    for i in range(0, N - window_len + 1, step):
        Cw = color_signal[:, i:i+window_len]
        Cn = Cw / np.mean(Cw, axis=1, keepdims=True)

        XY = chrom_matrix @ Cn
        stdev = np.std(XY, axis=1)
        alpha = stdev[0] / stdev[1]
        Sw = XY[0] + alpha * XY[1]
        Sw -= np.mean(Sw)
        S[i:i+window_len] += Sw
        W[i:i+window_len] += 1
    
    W = np.where(W == 0, 1e-6, W)
    S /= W
    S = scipy.signal.detrend(S)
    S = bandpass_filter(S, fps)
    if show_graph:
        plt.plot(S)
        plt.title("POS Windowed: 1D signal")
        plt.show()

    return S