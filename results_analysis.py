import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from plots import *
import pandas as pd
from great_tables import GT
from statsmodels.stats.anova import AnovaRM

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

# Bland-Altman LOA

print("\n BLAND ALTMAN LOA:")
mean_error = np.mean(errors, axis=0)
std_error = np.std(errors, axis=0)
high_LOA = mean_error + 1.96 * std_error
low_LOA = mean_error - 1.96 * std_error
print(f"{high_LOA=}\n{low_LOA=}")

# MAE

stat_data = {}
subject_mae = []
subject_rmse = []
subject_bias = []
subject_r = []

for s in unique_subjects:
    idx = subject == s
    subject_mae.append(np.mean(abs_errors[idx], axis=0))
    subject_rmse.append(np.sqrt(np.mean(np.square(errors[idx]), axis=0)))
    subject_bias.append(np.mean(errors[idx], axis=0))
    subject_r.append([stats.pearsonr(rppg[:, i], gt_hr)[0] for i in range(rppg.shape[1])])

def display_overall_using_subject(subject_data, name, stat_dict):
    print(f"========={name}=========")
    subject_data = np.array(subject_data)

    mean = np.mean(subject_data, axis=0)
    median = np.median(subject_data, axis=0)
    std = np.std(subject_data, axis=0)

    stat_dict[f"{name}-mean"] = mean
    stat_dict[f"{name}-median"] = median
    stat_dict[f"{name}-std"] = std

    print(f"{mean=}")
    print(f"{median=}")
    print(f"{std=}")

display_overall_using_subject(subject_mae, "MAE", stat_data)
display_overall_using_subject(subject_rmse, "RMSE", stat_data)
display_overall_using_subject(subject_bias, "BIAS", stat_data)
display_overall_using_subject(subject_r, "Pearson R", stat_data)

within_5 = np.mean(abs_errors <= 5, axis=0) * 100
within_10 = np.mean(abs_errors <= 10, axis=0) * 100

print("% within ±5 bpm:", within_5)
print("% within ±10 bpm:", within_10)

wilcoxon = [stats.wilcoxon(errors[:, i][errors[:, i] != 0]) for i in range(rppg.shape[1])]

wilcoxon_stats = [w.statistic for w in wilcoxon]
wilcoxon_p = [w.pvalue for w in wilcoxon]

print(wilcoxon_stats)
print()
print(wilcoxon_p)
print("This shows there is a statistically significant bias. Earlier we observed that BIAS < 0, so all rPPG methods underestimate.")

method_names = [
    "GREEN",
    "GREEN/RED",
    "GREEN/BLUE",
    "CHROM",
    "POS"
]

methods = [f"{method} {window} ({pp})" for method in method_names for window in ("", "windowed") for pp in ("rFFT", "Periodogram", "Peak")]

full_stat_df = pd.DataFrame({
    "Method": methods,
    "B-A LOA-high": high_LOA,
    "B-A LOA-low": low_LOA,
    "MAE": stat_data["MAE-mean"],
    "RMSE": stat_data["RMSE-mean"],
    "Bias": stat_data["BIAS-mean"],
    "Error Std per Subject": stat_data["BIAS-std"],
    "Pearson R": stat_data["Pearson R-mean"],
    "Wilcoxon P": wilcoxon_p,
    "Wilcoxon T": wilcoxon_stats,
    "% within ±5 bpm:": within_5,
    "% within ±10 bpm:": within_10,
})

print(full_stat_df)

df_sorted = full_stat_df.sort_values(by="MAE")
print(df_sorted)

# table = GT(df).tab_header(title=f"rPPG Statistics on UBFC Dataset, n={total-num_bad}")
# table.save("stats.png")

subject_level_error = np.array(subject_mae)

n_subjects, n_methods = subject_level_error.shape
print(len(methods))

records = []

for subj in range(n_subjects):
    for j, name in enumerate(methods):
        base, rest = name.split(" ", 1)
        window = "windowed" if "windowed" in rest else "none"
        peak = rest.split("(")[1].strip(")")

        records.append({
            "subject": subj,
            "rppg": base,
            "window": window,
            "hr_method": peak,
            "error": subject_level_error[subj, j]
        })

df = pd.DataFrame(records)
print(df)

anova = AnovaRM(
    df,
    depvar="error",
    subject="subject",
    within=["rppg", "window", "hr_method"]
).fit()

print(anova)

# Effect of windowing within each (rPPG, hr_method)

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

rows = []
pvals = []
labels = []

for rppg in df["rppg"].unique():
    for hr in df["hr_method"].unique():
        sub = df[(df.rppg == rppg) & (df.hr_method == hr)]
        w0 = sub[sub.window == "none"].error.values
        w1 = sub[sub.window == "windowed"].error.values

        diff = w1 - w0
        t, p = ttest_rel(w0, w1)
        pvals.append(p)
        labels.append(f"{rppg} | {hr}: window vs no-window")
        rows.append({
            "rppg": rppg,
            "hr_method": hr,
            "mean_diff": diff.mean(),
            "p": p
        })


res = pd.DataFrame(rows)
reject, p_corr, _, _ = multipletests(pvals, method="holm")

results = list(zip(labels, p_corr, reject))
# print(results)

res["p_corr"] = multipletests(res.p, method="holm")[1]
res["sig"] = res.p_corr < 0.05
print("mean_diff represents windowed - non-windowed")
print(res)

heat = res.pivot(index="rppg", columns="hr_method", values="mean_diff")
sig  = res.pivot(index="rppg", columns="hr_method", values="sig")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 4))

sns.heatmap(
    heat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Δ HR (windowed − no-window)"},
    mask=~sig  # hide non-significant cells
)

plt.title("Effect of Windowing on HR Estimation\n(significant only, Holm-corrected)")
plt.ylabel("rPPG method")
plt.xlabel("HR derivation method")
plt.tight_layout()
# plt.show()

plt.figure(figsize=(6, 6))

sns.pointplot(
    data=res,
    y="rppg",
    x="mean_diff",
    hue="hr_method",
    dodge=True,
    linestyles='none',
    errorbar=('ci', 95)
)

plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Δ HR (windowed − no-window)")
plt.title("Windowing effect by rPPG and HR method")
plt.tight_layout()
# plt.show()


# 

from itertools import combinations

pvals, labels = [], []
rows = []

for rppg in df.rppg.unique():
    for window in df.window.unique():
        sub = df[(df.rppg == rppg) & (df.window == window)]

        for a, b in combinations(sub.hr_method.unique(), 2):
            x = sub[sub.hr_method == a].error.values
            y = sub[sub.hr_method == b].error.values
            diff = x - y
            _, p = ttest_rel(x, y)

            pvals.append(p)
            labels.append(f"{rppg} | {window}: {a} vs {b}")
            rows.append({
                "rppg": rppg,
                "window": window,
                "comparison": f"{a} - {b}",
                "mean_diff": diff.mean(),
                "p": p
            })

res = pd.DataFrame(rows)
# Holm correction within this family
res["p_corr"] = multipletests(res.p, method="holm")[1]
res["sig"] = res.p_corr < 0.05
reject, p_corr, _, _ = multipletests(pvals, method="holm")
results = list(zip(labels, p_corr, reject))
print("\nEffect of HR-derivation method within each (rPPG, window)")
# print(results)
print(res)

def pivot_for_window(win):
    sub = res[res.window == win]
    heat = sub.pivot(index="rppg", columns="comparison", values="mean_diff")
    sig  = sub.pivot(index="rppg", columns="comparison", values="sig")
    return heat, sig

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, win in zip(axes, ["none", "windowed"]):
    heat, sig = pivot_for_window(win)

    sns.heatmap(
        heat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        mask=~sig,
        cbar=ax is axes[1],
        cbar_kws={"label": "Δ Absolute Error (bpm)"}
    )

    ax.set_title(f"HR-method comparison ({win})")
    ax.set_xlabel("HR-method contrast")
    ax.set_ylabel("rPPG method")

plt.suptitle(
    "Effect of HR-derivation method within each (rPPG × window)\n"
    "Cells shown only if Holm-corrected p < 0.05",
    y=0.99
)

plt.tight_layout()
# plt.show()

# Compute MAE per (rppg, window, hr_method)
config_mae = (
    df.groupby(["rppg", "window", "hr_method"])["error"]
      .mean()
      .reset_index()
      .rename(columns={"error": "mae"})
)

# Find best configuration
best_cfg = config_mae.loc[config_mae.mae.idxmin()]
print(best_cfg)
top_cfgs = config_mae.nsmallest(3, "mae")
print(top_cfgs)


best_cfgs = (
    df.groupby(["rppg", "window", "hr_method"])["error"]
      .mean()
      .reset_index()
      .sort_values("error")
      .groupby("rppg")
      .first()
)

print("===BEST CONFIGURATIONS FOR EACH rPPG===")
print(best_cfgs)

best_rows = []

for _, row in best_cfgs.iterrows():
    print('row:')
    print(row)
    sub = df[
        (df.rppg == row.name) &
        (df.window == row.window) &
        (df.hr_method == row.hr_method)
    ]
    best_rows.append(sub)

best_df = pd.concat(best_rows)

print(best_df) # 39 subjects * 5 configurations

pvals = []

pivot = best_df.pivot(index="subject", columns="rppg", values="error")
rows = []
for a, b in combinations(pivot.columns, 2):
    x = pivot[a]
    y = pivot[b]
    diff = x-y
    _, p = ttest_rel(x, y)
    pvals.append(p)
    rows.append({
        "comparison": f"{a} - {b}",
        "mean_diff": diff.mean(),
        "p": p
    })

reject, p_corr, _, _ = multipletests(pvals, method="holm")
res = pd.DataFrame(rows)
res["p_corr"] = multipletests(res.p, method="holm")[1]
res["sig"] = res.p_corr < 0.05
print(res)