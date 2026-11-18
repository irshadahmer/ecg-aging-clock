import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet(os.path.join("results", "test_analyzed.parquet"))
or_df = pd.read_csv(os.path.join("results", "tables", "table3_clinical_odds_ratios.csv"))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

# Panel A: Predicted vs Chronological Age (12-Lead)
ax = axes[0]
ax.scatter(df["age"], df["corr_age_12lead"], alpha=0.22, s=12, color="#1f77b4", edgecolors="none")
lims = [18, 95]
ax.plot(lims, lims, "--", color="#333333", linewidth=1.3, label="Identity (y = x)")
ax.set_xlim(lims)
ax.set_ylim(10, 105)
ax.set_xlabel("Chronological Age (years)")
ax.set_ylabel("Bias-Corrected 12-Lead ECG Age (years)")
ax.set_title("A. 12-Lead Chronological Tracking (r = 0.81)", fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper left")

# Panel B: Residualized Delta-Age vs Chronological Age
ax = axes[1]
ax.scatter(df["age"], df["delta_age_12lead"], alpha=0.22, s=12, color="#2ca02c", edgecolors="none")
ax.axhline(0, linestyle="--", color="#333333", linewidth=1.3)
ax.axhline(5, linestyle=":", color="#d62728", linewidth=1.1, label="Accelerated (> +5 yrs)")
ax.set_xlim(lims)
ax.set_xlabel("Chronological Age (years)")
ax.set_ylabel("ECG-Age Gap, ΔAge (years)")
ax.set_title("B. Age-Bias Neutralization (r = -0.04)", fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper right")

# Panel C: Clinical Odds Ratio Forest Plot
ax = axes[2]
conds = ["Myocardial Infarction (MI)", "ST/T Changes (STTC)", "Conduction Disturbance (CD)", "Hypertrophy (HYP)", "Normal ECG (NORM)"]
short_labels = ["MI", "ST/T Changes", "Conduction Dist.", "Hypertrophy", "Normal ECG"]
y_pos = np.arange(len(conds))

for offset, model_name, color, marker in [(-0.15, "12-Lead", "#1f77b4", "o"), (0.15, "Lead-I", "#ff7f0e", "s")]:
    sub = or_df[or_df["model"] == model_name].set_index("condition").loc[conds]
    ors = sub["odds_ratio"].values
    ci_lows = np.array([float(x.split("-")[0]) for x in sub["ci_95"]])
    ci_highs = np.array([float(x.split("-")[1]) for x in sub["ci_95"]])
    ax.errorbar(
        ors, y_pos + offset, xerr=[ors - ci_lows, ci_highs - ors],
        fmt=marker, color=color, ecolor=color, elinewidth=1.6, capsize=3, markersize=6, label=model_name
    )

ax.axvline(1.0, linestyle="--", color="#333333", linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(short_labels)
ax.invert_yaxis()
ax.set_xlabel("Adjusted Odds Ratio (95% CI)")
ax.set_title("C. Disease Odds for ΔAge > +5 yrs", fontweight="bold", loc="left")
ax.legend(frameon=False, loc="lower right")

plt.tight_layout()
out_path = os.path.join("results", "figures", "fig1_ecg_aging_clock.png")
plt.savefig(out_path, dpi=300)
print(f"Saved publication figure to {out_path}")
