import numpy as np
from pathlib import Path
from scipy import stats

# Specify the directory path
results_dir = Path("results")

data = np.loadtxt(results_dir / 'concatenated_results.csv', delimiter=',')

# Analyze the concatenated data

# First check difference between ground truth heart rate estimates 
gt_avg_hr = data[:, 1]
gt_ppg_peak = data[:, 2]
gt_ppg_fft = data[:, 3]

bad_gt = (
    (gt_avg_hr < 30) |
    (gt_ppg_peak < 30) |
    (gt_ppg_fft < 30) |
    (np.abs(gt_ppg_peak - gt_ppg_fft) > 20)
)

# bad_ppg = ((gt_ppg_peak < 30) | (gt_ppg_fft < 30))
# print(np.unique(data[bad_ppg][:, 0]))
print("Potential Errors in Ground Truth Values:")
print(np.unique(data[bad_gt][:, 0]))
print()

data = data[~bad_gt]
gt_avg_hr = data[:, 1]
gt_ppg_peak = data[:, 2]
gt_ppg_fft = data[:, 3]

gt_methods = {
    "Wearable Avg HR": gt_avg_hr,
    "Wearable Peak HR": gt_ppg_peak,
    "Wearable FFT HR": gt_ppg_fft,
}

# ==============================
# BASIC DESCRIPTIVE STATISTICS
# ==============================

print("\n==============================")
print("GROUND TRUTH DESCRIPTIVE STATS")
print("==============================\n")

for name, hr in gt_methods.items():
    print(f"{name:20s} | "
          f"Mean: {np.mean(hr):6.2f} bpm | "
          f"Std: {np.std(hr):6.2f} bpm | "
          f"Median: {np.median(hr):6.2f} bpm | "
          f"IQR: {(np.percentile(hr, 75) - np.percentile(hr, 25)):6.2f}")

# ==============================
# PAIRWISE AGREEMENT ANALYSIS
# ==============================

print("\n==============================")
print("PAIRWISE GROUND TRUTH AGREEMENT")
print("==============================\n")

pairs = [
    ("Avg HR", gt_avg_hr, "Peak HR", gt_ppg_peak),
    ("Avg HR", gt_avg_hr, "FFT HR", gt_ppg_fft),
    ("Peak HR", gt_ppg_peak, "FFT HR", gt_ppg_fft),
]

all_abs_diffs = []

for name_a, a, name_b, b in pairs:
    diff = a - b
    abs_diff = np.abs(diff)
    all_abs_diffs.append(abs_diff)

    print(f"{name_a} vs {name_b}")
    print("-" * (len(name_a) + len(name_b) + 4))
    print(f"Mean difference (bias): {np.mean(diff):6.2f} bpm")
    print(f"Std of difference     : {np.std(diff):6.2f} bpm")
    print(f"Median abs difference: {np.median(abs_diff):6.2f} bpm")

    # Paired statistical test
    t_stat, p_val = stats.ttest_rel(a, b, nan_policy="omit")
    print(f"Paired t-test p-value : {p_val:.4e}\n")

# ==============================
# NOISE FLOOR ESTIMATE
# ==============================

all_abs_diffs = np.concatenate(all_abs_diffs)

print("\n==============================")
print("GROUND TRUTH NOISE FLOOR")
print("==============================\n")

print(f"Median absolute GT disagreement : {np.median(all_abs_diffs):.2f} bpm")
print(f"90th percentile disagreement   : {np.percentile(all_abs_diffs, 90):.2f} bpm")
print(f"95th percentile disagreement   : {np.percentile(all_abs_diffs, 95):.2f} bpm")

print("\nInterpretation:")
print("- This disagreement represents the lower bound on achievable rPPG error.")
print("- rPPG MAE values near or below this range approach GT measurement limits.\n")

# from this, we notice that there are errors in the ground truth heart rate value displayed by the wearable
# -> we need will use the heart rate values derived from the PPG signal as the ground truth

ppg_disagreement = (np.abs(gt_ppg_peak - gt_ppg_fft) > 5)

num_bad = np.sum(ppg_disagreement)
total = len(ppg_disagreement)

print(f"Excluded windows: {num_bad}/{total} "
      f"({100 * num_bad / total:.2f}%)")

gt_hr = np.median(
    np.vstack([gt_ppg_peak, gt_ppg_fft]),
    axis=0
)
gt_hr = gt_hr[~ppg_disagreement]
data = data[~ppg_disagreement]

subject = data[:, 0].astype(int)
rppg = data[:, 4:]   # shape (N, num_models)
errors = rppg - gt_hr[:, None]
abs_errors = np.abs(errors)

unique_subjects = np.unique(subject)

subject_mae = []

for s in unique_subjects:
    idx = subject == s
    subject_mae.append(np.mean(abs_errors[idx], axis=0))

subject_mae = np.array(subject_mae)

mean_mae = np.mean(subject_mae, axis=0)
median_mae = np.median(subject_mae, axis=0)
std_mae = np.std(subject_mae, axis=0)

print(f"{mean_mae=}")
print(f"{median_mae=}")
print(f"{std_mae=}")