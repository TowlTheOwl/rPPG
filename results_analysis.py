"""
rPPG Pipeline Evaluation
========================

Analyzes heart-rate estimates produced by several rPPG pipelines against
wearable/PPG-derived ground truth, over ~40 subjects, using 10s sliding
windows (1s step).

Input: results/concatenated_results.csv
Columns:
    0  subject id
    1  ground truth HR from wearable device (found to be unreliable)
    2  ground truth HR from PPG signal, peak-detection method
    3  ground truth HR from PPG signal, FFT method
    4+ HR estimates from each rPPG pipeline / HR-derivation method combo

Pipeline of this script:
    1. Load data, drop windows with clearly broken ground truth
    2. Compare the three GT sources, establish a GT noise floor
    3. Build a consensus GT HR from the two PPG-derived methods
    4. Compute per-method error statistics (Bland-Altman, MAE, RMSE, bias,
       Pearson r, Wilcoxon signed-rank test)
    5. Repeated-measures ANOVA across (rPPG method x window x HR-derivation
       method), with pairwise follow-up tests (Holm-corrected)
    6. Identify the best-performing configuration per rPPG method and
       compare rPPG methods head-to-head using their best configs
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from great_tables import GT

from plots import *  # noqa: F401,F403 (project plotting helpers)

RESULTS_DIR = Path("results")
DATA_FILE = RESULTS_DIR / "concatenated_results.csv"

# GT sanity-check thresholds
MIN_PLAUSIBLE_HR = 30          # bpm; anything below is a sensor glitch
MAX_PPG_GT_DISAGREEMENT = 20   # bpm; peak vs FFT GT sanity bound
PPG_GT_EXCLUSION_THRESHOLD = 5  # bpm; peak vs FFT disagreement -> drop window

METHOD_NAMES = ["GREEN", "GREEN/RED", "GREEN/BLUE", "CHROM", "POS"]
WINDOW_VARIANTS = ("", "windowed")
HR_DERIVATION_VARIANTS = ("rFFT", "Periodogram", "Peak")

SECTION_WIDTH = 70


# --------------------------------------------------------------------------
# Console output helpers (kept consistent across the whole script)
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * SECTION_WIDTH)
    print(title.upper())
    print("=" * SECTION_WIDTH)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def stat(label: str, value, fmt: str = "6.2f", unit: str = "bpm") -> None:
    print(f"  {label:<28}: {value:{fmt}} {unit}")


# 1. Load & clean data

def load_data(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def drop_bad_ground_truth(data: np.ndarray) -> np.ndarray:
    """Remove windows where any GT source looks like a sensor glitch, or
    where the two PPG-derived GT estimates disagree wildly."""
    gt_avg_hr, gt_peak, gt_fft = data[:, 1], data[:, 2], data[:, 3]

    bad_gt = (
        (gt_avg_hr < MIN_PLAUSIBLE_HR)
        | (gt_peak < MIN_PLAUSIBLE_HR)
        | (gt_fft < MIN_PLAUSIBLE_HR)
        | (np.abs(gt_peak - gt_fft) > MAX_PPG_GT_DISAGREEMENT)
    )

    section("Ground truth quality check")
    print("Subjects with at least one implausible GT window:")
    print(np.unique(data[bad_gt][:, 0]))

    return data[~bad_gt]


# 2. Ground-truth agreement analysis

def describe_ground_truth(gt_methods: dict) -> None:
    section("Ground truth descriptive statistics")
    for name, hr in gt_methods.items():
        subsection(name)
        stat("Mean", np.mean(hr))
        stat("Std", np.std(hr))
        stat("Median", np.median(hr))
        stat("IQR", np.percentile(hr, 75) - np.percentile(hr, 25))


def pairwise_gt_agreement(gt_avg_hr, gt_peak, gt_fft) -> np.ndarray:
    """Compare each pair of GT sources; returns concatenated abs differences
    for the noise-floor estimate."""
    section("Pairwise ground truth agreement")

    pairs = [
        ("Avg HR", gt_avg_hr, "Peak HR", gt_peak),
        ("Avg HR", gt_avg_hr, "FFT HR", gt_fft),
        ("Peak HR", gt_peak, "FFT HR", gt_fft),
    ]

    all_abs_diffs = []
    for name_a, a, name_b, b in pairs:
        diff = a - b
        abs_diff = np.abs(diff)
        all_abs_diffs.append(abs_diff)

        subsection(f"{name_a} vs {name_b}")
        stat("Mean difference (bias)", np.mean(diff))
        stat("Std of difference", np.std(diff))
        stat("Median abs difference", np.median(abs_diff))

        _, p_val = stats.ttest_rel(a, b, nan_policy="omit")
        stat("Paired t-test p-value", p_val, fmt=".4e", unit="")

    return np.concatenate(all_abs_diffs)


def report_noise_floor(all_abs_diffs: np.ndarray) -> None:
    section("Ground truth noise floor")
    stat("Median absolute GT disagreement", np.median(all_abs_diffs))
    stat("90th percentile disagreement", np.percentile(all_abs_diffs, 90))
    stat("95th percentile disagreement", np.percentile(all_abs_diffs, 95))
    print(
        "\nInterpretation: this disagreement is the lower bound on "
        "achievable rPPG error. rPPG MAE values near or below this range "
        "approach GT measurement limits."
    )


# 3. Consensus ground truth (wearable HR is unreliable -> use PPG-derived)

def build_consensus_gt(data: np.ndarray):
    """Median of peak/FFT PPG-derived HR, excluding windows where the two
    disagree by more than PPG_GT_EXCLUSION_THRESHOLD bpm."""
    gt_peak, gt_fft = data[:, 2], data[:, 3]
    ppg_disagreement = np.abs(gt_peak - gt_fft) > PPG_GT_EXCLUSION_THRESHOLD

    num_bad = np.sum(ppg_disagreement)
    total = len(ppg_disagreement)
    section("Consensus ground truth (PPG peak/FFT median)")
    print(
        f"Excluded windows: {num_bad}/{total} "
        f"({100 * num_bad / total:.2f}%)"
    )

    gt_hr = np.median(np.vstack([gt_peak, gt_fft]), axis=0)
    gt_hr = gt_hr[~ppg_disagreement]
    data = data[~ppg_disagreement]

    return data, gt_hr


# 4. Error metrics per method

def bland_altman_loa(errors: np.ndarray):
    section("Bland-Altman limits of agreement")
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    high_loa = mean_error + 1.96 * std_error
    low_loa = mean_error - 1.96 * std_error
    print(f"high_LOA = {high_loa}")
    print(f"low_LOA  = {low_loa}")
    return high_loa, low_loa


def subject_level_metrics(subject, rppg, errors, abs_errors, gt_hr):
    """Per-subject MAE / RMSE / bias / Pearson r, one row per subject."""
    unique_subjects = np.unique(subject)

    subject_mae, subject_rmse, subject_bias, subject_r = [], [], [], []

    for s in unique_subjects:
        idx = subject == s
        subject_mae.append(np.mean(abs_errors[idx], axis=0))
        subject_rmse.append(np.sqrt(np.mean(np.square(errors[idx]), axis=0)))
        subject_bias.append(np.mean(errors[idx], axis=0))
        subject_r.append(
            [stats.pearsonr(rppg[:, i], gt_hr)[0] for i in range(rppg.shape[1])]
        )

    return subject_mae, subject_rmse, subject_bias, subject_r


def summarize_across_subjects(subject_data, name, stat_dict) -> None:
    subsection(name)
    subject_data = np.array(subject_data)

    mean = np.mean(subject_data, axis=0)
    median = np.median(subject_data, axis=0)
    std = np.std(subject_data, axis=0)

    stat_dict[f"{name}-mean"] = mean
    stat_dict[f"{name}-median"] = median
    stat_dict[f"{name}-std"] = std

    print(f"  mean   = {mean}")
    print(f"  median = {median}")
    print(f"  std    = {std}")


def within_tolerance(abs_errors: np.ndarray):
    within_5 = np.mean(abs_errors <= 5, axis=0) * 100
    within_10 = np.mean(abs_errors <= 10, axis=0) * 100

    section("Accuracy within tolerance bands")
    print("% within +/-5 bpm :", within_5)
    print("% within +/-10 bpm:", within_10)
    return within_5, within_10


def wilcoxon_bias_test(errors: np.ndarray):
    section("Wilcoxon signed-rank test (per-method bias)")
    results = [
        stats.wilcoxon(errors[:, i][errors[:, i] != 0])
        for i in range(errors.shape[1])
    ]
    wilcoxon_stats = [w.statistic for w in results]
    wilcoxon_p = [w.pvalue for w in results]

    print("Statistics:", wilcoxon_stats)
    print("P-values:  ", wilcoxon_p)
    print(
        "All p-values below indicate a statistically significant bias; "
        "since bias < 0 (see above), all rPPG methods underestimate HR."
    )
    return wilcoxon_stats, wilcoxon_p


# 5. Summary table across all method/window/HR-derivation combinations

def build_method_labels():
    return [
        f"{method} {window} ({pp})"
        for method in METHOD_NAMES
        for window in WINDOW_VARIANTS
        for pp in HR_DERIVATION_VARIANTS
    ]


def build_summary_table(methods, high_loa, low_loa, stat_data, wilcoxon_p,
                         wilcoxon_stats, within_5, within_10) -> pd.DataFrame:
    df = pd.DataFrame({
        "Method": methods,
        "MAE": stat_data["MAE-mean"],
        "RMSE": stat_data["RMSE-mean"],
        "Bias": stat_data["BIAS-mean"],
        "Error Std per Subject": stat_data["BIAS-std"],
        "Pearson R": stat_data["Pearson R-mean"],
        "Wilcoxon P": wilcoxon_p,
        "Wilcoxon T": wilcoxon_stats,
        "% within +/-5 bpm": within_5,
        "% within +/-10 bpm": within_10,
        "B-A LOA-high": high_loa,
        "B-A LOA-low": low_loa,
    })

    section("Full summary table (all method/window/HR-derivation combos)")
    print(df)

    subsection("Sorted by MAE")
    print(df.sort_values(by="MAE"))

    subsection("Sroted by % within 5 bpm")
    print(df.sort_values(by="% within +/-5 bpm", ascending=False))

    return df


# 6. Long-format dataframe for ANOVA / follow-up tests

def parse_method_label(name: str):
    base, rest = name.split(" ", 1)
    window = "windowed" if "windowed" in rest else "none"
    hr_method = rest.split("(")[1].strip(")")
    return base, window, hr_method


def build_long_dataframe(subject_mae, methods) -> pd.DataFrame:
    subject_level_error = np.array(subject_mae)
    n_subjects, n_methods = subject_level_error.shape

    records = []
    for subj in range(n_subjects):
        for j, name in enumerate(methods):
            base, window, hr_method = parse_method_label(name)
            records.append({
                "subject": subj,
                "rppg": base,
                "window": window,
                "hr_method": hr_method,
                "error": subject_level_error[subj, j],
            })

    df = pd.DataFrame(records)

    section("Long-format per-subject error table")
    print(df)
    return df


def run_repeated_measures_anova(df: pd.DataFrame):
    section("Repeated-measures ANOVA (rppg x window x hr_method)")
    anova = AnovaRM(
        df, depvar="error", subject="subject",
        within=["rppg", "window", "hr_method"],
    ).fit()
    print(anova)
    return anova


# 7. Follow-up: effect of windowing, within each (rPPG, HR-derivation)

def windowing_effect(df: pd.DataFrame) -> pd.DataFrame:
    section("Effect of windowing within each (rPPG, HR-derivation method)")

    rows, pvals = [], []
    for rppg in df["rppg"].unique():
        for hr in df["hr_method"].unique():
            sub = df[(df.rppg == rppg) & (df.hr_method == hr)]
            w0 = sub[sub.window == "none"].error.values
            w1 = sub[sub.window == "windowed"].error.values

            diff = w1 - w0
            _, p = ttest_rel(w0, w1)
            pvals.append(p)
            rows.append({
                "rppg": rppg, "hr_method": hr,
                "mean_diff": diff.mean(), "p": p,
            })

    res = pd.DataFrame(rows)
    res["p_corr"] = multipletests(res.p, method="holm")[1]
    res["sig"] = res.p_corr < 0.05

    print("mean_diff = windowed - non-windowed")
    print(res)
    return res


def plot_windowing_effect(res: pd.DataFrame) -> None:
    heat = res.pivot(index="rppg", columns="hr_method", values="mean_diff")
    sig = res.pivot(index="rppg", columns="hr_method", values="sig")

    plt.figure(figsize=(7, 4))
    sns.heatmap(
        heat, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        cbar_kws={"label": "\u0394 HR (windowed - no-window)"}, mask=~sig,
    )
    plt.title(
        "Effect of Windowing on HR Estimation\n"
        "(significant only, Holm-corrected)"
    )
    plt.ylabel("rPPG method")
    plt.xlabel("HR derivation method")
    plt.tight_layout()

    plt.figure(figsize=(6, 6))
    sns.pointplot(
        data=res, y="rppg", x="mean_diff", hue="hr_method",
        dodge=True, linestyles="none", errorbar=("ci", 95),
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("\u0394 HR (windowed - no-window)")
    plt.title("Windowing effect by rPPG and HR method")
    plt.tight_layout()


# 8. Follow-up: effect of HR-derivation method within each (rPPG, window)

def hr_method_effect(df: pd.DataFrame) -> pd.DataFrame:
    section("Effect of HR-derivation method within each (rPPG, window)")

    rows, pvals = [], []
    for rppg in df.rppg.unique():
        for window in df.window.unique():
            sub = df[(df.rppg == rppg) & (df.window == window)]

            for a, b in combinations(sub.hr_method.unique(), 2):
                x = sub[sub.hr_method == a].error.values
                y = sub[sub.hr_method == b].error.values
                diff = x - y
                _, p = ttest_rel(x, y)

                pvals.append(p)
                rows.append({
                    "rppg": rppg, "window": window,
                    "comparison": f"{a} - {b}",
                    "mean_diff": diff.mean(), "p": p,
                })

    res = pd.DataFrame(rows)
    res["p_corr"] = multipletests(res.p, method="holm")[1]
    res["sig"] = res.p_corr < 0.05
    print(res)
    return res


def plot_hr_method_effect(res: pd.DataFrame) -> None:
    def pivot_for_window(win):
        sub = res[res.window == win]
        heat = sub.pivot(index="rppg", columns="comparison", values="mean_diff")
        sig = sub.pivot(index="rppg", columns="comparison", values="sig")
        return heat, sig

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, win in zip(axes, ["none", "windowed"]):
        heat, sig = pivot_for_window(win)
        sns.heatmap(
            heat, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            mask=~sig, cbar=ax is axes[1],
            cbar_kws={"label": "\u0394 Absolute Error (bpm)"},
        )
        ax.set_title(f"HR-method comparison ({win})")
        ax.set_xlabel("HR-method contrast")
        ax.set_ylabel("rPPG method")

    plt.suptitle(
        "Effect of HR-derivation method within each (rPPG x window)\n"
        "Cells shown only if Holm-corrected p < 0.05",
        y=0.99,
    )
    plt.tight_layout()


# 9. Best configuration per rPPG method, then head-to-head comparison

def best_configurations(df: pd.DataFrame):
    config_mae = (
        df.groupby(["rppg", "window", "hr_method"])["error"]
        .mean()
        .reset_index()
        .rename(columns={"error": "mae"})
    )

    section("Best configuration overall")
    print(config_mae.loc[config_mae.mae.idxmin()])

    subsection("Top 3 configurations overall")
    print(config_mae.nsmallest(3, "mae"))

    best_cfgs = (
        df.groupby(["rppg", "window", "hr_method"])["error"]
        .mean()
        .reset_index()
        .sort_values("error")
        .groupby("rppg")
        .first()
    )

    subsection("Best configuration per rPPG method")
    print(best_cfgs)

    return best_cfgs


def best_config_subject_data(df: pd.DataFrame, best_cfgs: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for rppg_name, row in best_cfgs.iterrows():
        sub = df[
            (df.rppg == rppg_name)
            & (df.window == row.window)
            & (df.hr_method == row.hr_method)
        ]
        best_rows.append(sub)

    best_df = pd.concat(best_rows)

    section("Per-subject error using each rPPG method's best configuration")
    print(best_df)  # n_subjects * n_rppg_methods rows
    return best_df


def compare_best_configs(best_df: pd.DataFrame) -> pd.DataFrame:
    pivot = best_df.pivot(index="subject", columns="rppg", values="error")

    rows, pvals = [], []
    for a, b in combinations(pivot.columns, 2):
        x, y = pivot[a], pivot[b]
        diff = x - y
        _, p = ttest_rel(x, y)
        pvals.append(p)
        rows.append({"comparison": f"{a} - {b}", "mean_diff": diff.mean(), "p": p})

    res = pd.DataFrame(rows)
    res["p_corr"] = multipletests(res.p, method="holm")[1]
    res["sig"] = res.p_corr < 0.05

    section("Head-to-head comparison of rPPG methods (best config each)")
    print(res)
    return res


def main() -> None:
    data = load_data(DATA_FILE)
    data = drop_bad_ground_truth(data)

    gt_avg_hr, gt_peak, gt_fft = data[:, 1], data[:, 2], data[:, 3]
    gt_methods = {
        "Wearable Avg HR": gt_avg_hr,
        "Wearable Peak HR": gt_peak,
        "Wearable FFT HR": gt_fft,
    }
    describe_ground_truth(gt_methods)

    all_abs_diffs = pairwise_gt_agreement(gt_avg_hr, gt_peak, gt_fft)
    report_noise_floor(all_abs_diffs)

    # Wearable HR is unreliable -> fall back to PPG-derived consensus GT
    data, gt_hr = build_consensus_gt(data)

    subject = data[:, 0].astype(int)
    rppg = data[:, 4:]
    errors = rppg - gt_hr[:, None]
    abs_errors = np.abs(errors)

    high_loa, low_loa = bland_altman_loa(errors)

    stat_data = {}
    subject_mae, subject_rmse, subject_bias, subject_r = subject_level_metrics(
        subject, rppg, errors, abs_errors, gt_hr
    )

    section("Per-method error statistics (aggregated across subjects)")
    summarize_across_subjects(subject_mae, "MAE", stat_data)
    summarize_across_subjects(subject_rmse, "RMSE", stat_data)
    summarize_across_subjects(subject_bias, "BIAS", stat_data)
    summarize_across_subjects(subject_r, "Pearson R", stat_data)

    within_5, within_10 = within_tolerance(abs_errors)
    wilcoxon_stats, wilcoxon_p = wilcoxon_bias_test(errors)

    methods = build_method_labels()
    summary_table = build_summary_table(
        methods, high_loa, low_loa, stat_data, wilcoxon_p, wilcoxon_stats,
        within_5, within_10,
    )
    table = GT(summary_table).tab_header(title=f"rPPG Statistics on UBFC Dataset, n={len(gt_hr)}")
    table.save("stat_summary.png")

    df = build_long_dataframe(subject_mae, methods)
    run_repeated_measures_anova(df)

    window_res = windowing_effect(df)
    plot_windowing_effect(window_res)

    hr_res = hr_method_effect(df)
    plot_hr_method_effect(hr_res)

    best_cfgs = best_configurations(df)
    best_df = best_config_subject_data(df, best_cfgs)
    compare_best_configs(best_df)

    # plt.show()  # uncomment to display figures interactively


if __name__ == "__main__":
    main()