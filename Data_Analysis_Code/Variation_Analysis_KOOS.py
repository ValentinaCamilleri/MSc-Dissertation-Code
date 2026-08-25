import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# 1. LOAD PARTICIPANT DATA
# ============================================================

EXCEL_FILE = r""

df = pd.read_excel(EXCEL_FILE)

# Remove any completely blank Excel columns such as "Unnamed: 24"
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]


# ============================================================
# 2. COLUMN DEFINITIONS
# ============================================================

SEX_COL = "Sex"
AGE_COL = "Age"
BMI_COL = "BMI"

KOOS_COLS = {
    "Pain": "Pain",
    "Symptoms": "Symptoms",
    "ADL": "ADL",
    "Sport/Recreation": "Sport",
    "QoL": "QoL",
}

BIOMARKER_COLS = {
    "T2": "T2 Global",
    "MD": "MD Global",
    "FA": "FA Global",
}


# ============================================================
# 3. PREPARE VARIABLES
# ============================================================

# Dissertation coding:
# Male = 0
# Female = 1
df["Gender"] = (
    df[SEX_COL]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "M": 0,
        "F": 1
    })
)

# Check that all sex values were recognised
if df["Gender"].isna().any():
    bad_values = df.loc[df["Gender"].isna(), SEX_COL].unique()

    raise ValueError(
        f"Unrecognised values found in Sex column: {bad_values}"
    )


# Ensure regression variables are numeric
numeric_cols = (
    [AGE_COL, BMI_COL]
    + list(KOOS_COLS.values())
    + list(BIOMARKER_COLS.values())
)

df[numeric_cols] = df[numeric_cols].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 4. FIT KOOS MODELS
# ============================================================
#
# Separate model for every KOOS subscale:
#
# Biomarker ~ KOOS + Gender + Age + BMI
#
# KOOS is divided by 10 so the coefficient represents
# biomarker change per 10-point increase in KOOS.
# ============================================================

results = {}
all_results = []

for biomarker_name, biomarker_col in BIOMARKER_COLS.items():

    biomarker_results = {
        "beta": [],
        "lower": [],
        "upper": [],
        "p": []
    }

    print("\n" + "=" * 100)
    print(f"{biomarker_name} — ADJUSTED KOOS ASSOCIATIONS")
    print("Adjusted for Gender, Age and BMI")
    print("Effect expressed per 10-point increase in KOOS")
    print("=" * 100)

    for subscale_name, koos_col in KOOS_COLS.items():

        model_df = df[
            [
                biomarker_col,
                "Gender",
                AGE_COL,
                BMI_COL,
                koos_col
            ]
        ].dropna().copy()

        # 1 regression unit = 10 KOOS points
        model_df["KOOS_10"] = model_df[koos_col] / 10.0

        y = model_df[biomarker_col]

        X = model_df[
            [
                "KOOS_10",
                "Gender",
                AGE_COL,
                BMI_COL
            ]
        ]

        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        beta = model.params["KOOS_10"]
        p_value = model.pvalues["KOOS_10"]

        ci = model.conf_int().loc["KOOS_10"]

        lower_ci = ci.iloc[0]
        upper_ci = ci.iloc[1]

        biomarker_results["beta"].append(beta)
        biomarker_results["lower"].append(lower_ci)
        biomarker_results["upper"].append(upper_ci)
        biomarker_results["p"].append(p_value)

        all_results.append({
            "Biomarker": biomarker_name,
            "KOOS subscale": subscale_name,
            "N": len(model_df),
            "Beta per 10 points": beta,
            "95% CI lower": lower_ci,
            "95% CI upper": upper_ci,
            "p-value": p_value,
        })

        print(
            f"{subscale_name:<20} "
            f"| N = {len(model_df):2d} "
            f"| B = {beta:9.4f} "
            f"| 95% CI = {lower_ci:9.4f} to {upper_ci:9.4f} "
            f"| p = {p_value:.4f}"
        )

    results[biomarker_name] = {
        key: np.asarray(value, dtype=float)
        for key, value in biomarker_results.items()
    }


# ============================================================
# 5. COMPLETE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 110)
print("COMPLETE ADJUSTED KOOS REGRESSION RESULTS")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Beta per 10 points": "{:.4f}".format,
            "95% CI lower": "{:.4f}".format,
            "95% CI upper": "{:.4f}".format,
            "p-value": "{:.4f}".format,
        }
    )
)


# ============================================================
# 6. FOREST PLOT
# ============================================================

subscales = [
    "Pain",
    "Symptoms",
    "ADL",
    "Sport/Recreation",
    "QoL"
]

y_positions = np.arange(len(subscales))[::-1]

xlabels = {
    "T2": r"T$_2$ change per 10-point KOOS increase (ms)",

    "MD":
        r"MD change per 10-point KOOS increase "
        r"($\times 10^{-3}$ mm$^2$/s)",

    "FA":
        "FA change per 10-point KOOS increase"
}

outcomes = ["T2", "MD", "FA"]
panel_labels = ["(a)", "(b)", "(c)"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.2, 4.8)
)

for ax, outcome, panel in zip(
    axes,
    outcomes,
    panel_labels
):

    beta = results[outcome]["beta"]
    lower = results[outcome]["lower"]
    upper = results[outcome]["upper"]
    pvals = results[outcome]["p"]

    xerr = np.vstack([
        beta - lower,
        upper - beta
    ])

    ax.errorbar(
        beta,
        y_positions,
        xerr=xerr,
        fmt="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        capthick=1.0
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        color="black"
    )

    ax.set_yticks(y_positions)

    if outcome == "T2":
        ax.set_yticklabels(subscales)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel(xlabels[outcome])

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5
    )

    ax.text(
        -0.12,
        1.04,
        panel,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin, xmax = ax.get_xlim()
    width = xmax - xmin

    ax.set_xlim(
        xmin,
        xmax + width * 0.20
    )

    xmin, xmax = ax.get_xlim()

    for yi, p_value in zip(
        y_positions,
        pvals
    ):

        if p_value < 0.001:
            p_text = r"$p<0.001$"
        else:
            p_text = rf"$p={p_value:.3f}$"

        ax.text(
            xmax - 0.02 * (xmax - xmin),
            yi,
            p_text,
            va="center",
            ha="right",
            fontsize=8.5
        )


fig.suptitle(
    "Adjusted associations between KOOS subscales and global qMRI biomarkers",
    fontsize=12,
    y=1.02
)

plt.tight_layout()
plt.show()
