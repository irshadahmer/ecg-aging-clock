import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_parquet(os.path.join("results", "predictions.parquet"))
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

for prefix in ["12lead", "1lead"]:
    col = f"pred_age_{prefix}"
    reg = LinearRegression().fit(val_df[["age"]], val_df[col])
    slope, intercept = reg.coef_[0], reg.intercept_
    test_df[f"corr_age_{prefix}"] = (test_df[col] - intercept) / slope
    test_df[f"delta_age_{prefix}"] = test_df[f"corr_age_{prefix}"] - test_df["age"]

def bootstrap_metrics(sub_df, y_col, pred_col, delta_col, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    pids = sub_df["patient_id"].unique()
    grouped = {pid: g for pid, g in sub_df.groupby("patient_id")}
    res = {"mae": [], "rmse": [], "r2": [], "r": [], "age_bias_r": []}
    for _ in range(n_boot):
        sample_pids = rng.choice(pids, size=len(pids), replace=True)
        b = pd.concat([grouped[p] for p in sample_pids], ignore_index=True)
        y, yh, da = b[y_col].values, b[pred_col].values, b[delta_col].values
        res["mae"].append(mean_absolute_error(y, yh))
        res["rmse"].append(np.sqrt(mean_squared_error(y, yh)))
        res["r2"].append(r2_score(y, yh))
        res["r"].append(np.corrcoef(y, yh)[0, 1])
        res["age_bias_r"].append(np.corrcoef(y, da)[0, 1])
    return {
        k: f"{np.mean(v):.2f} ({np.percentile(v, 2.5):.2f}-{np.percentile(v, 97.5):.2f})"
        for k, v in res.items()
    }

rows = []
for label, prefix in [("12-Lead Teacher", "12lead"), ("Lead-I Distilled Student", "1lead")]:
    test_df[f"raw_delta_{prefix}"] = test_df[f"pred_age_{prefix}"] - test_df["age"]
    raw_bias = np.corrcoef(test_df["age"], test_df[f"raw_delta_{prefix}"])[0, 1]
    m = bootstrap_metrics(test_df, "age", f"corr_age_{prefix}", f"delta_age_{prefix}")
    m["model"] = label
    m["raw_bias_r"] = f"{raw_bias:.2f}"
    rows.append(m)

perf_table = pd.DataFrame(rows)[["model", "mae", "rmse", "r2", "r", "raw_bias_r", "age_bias_r"]]
perf_table.to_csv(os.path.join("results", "tables", "table1_model_performance.csv"), index=False)
print("\n--- TABLE 1: OVERALL TEST SET PERFORMANCE (95% CI) ---")
print(perf_table.to_string(index=False))

test_df["age_group"] = pd.cut(test_df["age"], bins=[17, 49, 69, 100], labels=["<50 yrs", "50-69 yrs", ">=70 yrs"])
test_df["sex_label"] = test_df["sex"].map({0: "Male", 1: "Female"})

sub_rows = []
for group_col in ["age_group", "sex_label"]:
    for grp, gdf in test_df.groupby(group_col, observed=True):
        m12 = bootstrap_metrics(gdf, "age", "corr_age_12lead", "delta_age_12lead", n_boot=500)
        m1 = bootstrap_metrics(gdf, "age", "corr_age_1lead", "delta_age_1lead", n_boot=500)
        sub_rows.append({
            "subgroup": str(grp),
            "n_ecgs": len(gdf),
            "mae_12lead": m12["mae"],
            "mae_1lead": m1["mae"],
        })

sub_table = pd.DataFrame(sub_rows)
sub_table.to_csv(os.path.join("results", "tables", "table2_subgroup_fairness.csv"), index=False)
print("\n--- TABLE 2: SUBGROUP FAIRNESS AUDIT ---")
print(sub_table.to_string(index=False))

or_rows = []
for prefix, mname in [("12lead", "12-Lead"), ("1lead", "Lead-I")]:
    test_df[f"accel_{prefix}"] = (test_df[f"delta_age_{prefix}"] > 5.0).astype(int)
    for diag, dname in [
        ("diag_MI", "Myocardial Infarction (MI)"),
        ("diag_CD", "Conduction Disturbance (CD)"),
        ("diag_HYP", "Hypertrophy (HYP)"),
        ("diag_STTC", "ST/T Changes (STTC)"),
        ("diag_NORM", "Normal ECG (NORM)"),
    ]:
        X_mat = sm.add_constant(test_df[[f"accel_{prefix}", "age", "sex"]])
        logit_mod = sm.Logit(test_df[diag], X_mat).fit(disp=False)
        coef = logit_mod.params[f"accel_{prefix}"]
        ci_low, ci_high = logit_mod.conf_int().loc[f"accel_{prefix}"]
        pval = logit_mod.pvalues[f"accel_{prefix}"]
        or_rows.append({
            "model": mname,
            "condition": dname,
            "odds_ratio": round(np.exp(coef), 2),
            "ci_95": f"{np.exp(ci_low):.2f}-{np.exp(ci_high):.2f}",
            "p_value": f"{pval:.4g}",
        })

or_table = pd.DataFrame(or_rows)
or_table.to_csv(os.path.join("results", "tables", "table3_clinical_odds_ratios.csv"), index=False)
print("\n--- TABLE 3: ADJUSTED ODDS RATIOS FOR ACCELERATED ECG-AGE (DELTA_AGE > +5 YRS) ---")
print(or_table.to_string(index=False))

test_df.to_parquet(os.path.join("results", "test_analyzed.parquet"), index=False)
