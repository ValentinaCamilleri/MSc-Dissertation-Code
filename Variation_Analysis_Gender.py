import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# =========================================================
# 1. LOAD PARTICIPANT DATA FROM EXCEL
# =========================================================

EXCEL_FILE = r"C:\Users\Valentina\Documents\Masters_Data.xlsx"

df = pd.read_excel(EXCEL_FILE)

# Convert Excel headings such as "T2 MFC" to "T2_MFC".
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

# Convert spreadsheet gender labels to regression coding:
# Male = 0, Female = 1
df["Gender"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"M": 0, "F": 1})
)

required_columns = [
    "Code",
    "Gender",
    "Age",
    "BMI",
    "T2_Global",
    "MD_Global",
    "FA_Global",
    "T2_MFC",
    "T2_LFC",
    "T2_MTC",
    "T2_LTC",
    "MD_MFC",
    "MD_LFC",
    "MD_MTC",
    "MD_LTC",
    "FA_MFC",
    "FA_LFC",
    "FA_MTC",
    "FA_LTC",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required Excel columns: "
        + ", ".join(missing_columns)
    )

if df["Gender"].isna().any():
    raise ValueError(
        "Sex column must contain only M or F."
    )


# =========================================================
# 2. FIT GLOBAL MULTIVARIABLE MODELS
# =========================================================

model_t2 = smf.ols(
    "T2_Global ~ Gender + Age + BMI",
    data=df
).fit()

model_md = smf.ols(
    "MD_Global ~ Gender + Age + BMI",
    data=df
).fit()

model_fa = smf.ols(
    "FA_Global ~ Gender + Age + BMI",
    data=df
).fit()


global_models = {
    "T2": model_t2,
    "MD": model_md,
    "FA": model_fa
}


# =========================================================
# 3. CREATE PARTICIPANT-LEVEL ADJUSTED VALUES
#
# Age and BMI effects removed.
# Gender-related variation retained.
# =========================================================

mean_age = df["Age"].mean()
mean_bmi = df["BMI"].mean()

def create_gender_adjusted_values(outcome, model):

    return (
        df[outcome]

        - model.params["Age"]
        * (df["Age"] - mean_age)

        - model.params["BMI"]
        * (df["BMI"] - mean_bmi)

    )


df["T2_Adjusted_Gender"] = create_gender_adjusted_values(
    "T2_Global",
    model_t2
)

df["MD_Adjusted_Gender"] = create_gender_adjusted_values(
    "MD_Global",
    model_md
)

df["FA_Adjusted_Gender"] = create_gender_adjusted_values(
    "FA_Global",
    model_fa
)

# =========================================================
# 4. GLOBAL RESULTS TABLE
# =========================================================

global_rows = []

for biomarker, model in global_models.items():

    outcome = f"{biomarker}_Global"

    adjusted_values = create_gender_adjusted_values(
        outcome,
        model
    )

    female_values = adjusted_values[df["Gender"] == 1]
    male_values = adjusted_values[df["Gender"] == 0]

    global_rows.append({
        "Biomarker": biomarker,
        "Female mean": female_values.mean(),
        "Female SD": female_values.std(ddof=1),
        "Male mean": male_values.mean(),
        "Male SD": male_values.std(ddof=1),
        "Female - Male": model.params["Gender"],
        "p-value": model.pvalues["Gender"]
    })

global_table = pd.DataFrame(global_rows)

print("\nGLOBAL GENDER RESULTS")
print(
    global_table.to_string(
        index=False,
        formatters={
            "Female mean": "{:.3f}".format,
            "Female SD": "{:.3f}".format,
            "Male mean": "{:.3f}".format,
            "Male SD": "{:.3f}".format,
            "Female - Male": "{:.3f}".format,
            "p-value": "{:.3f}".format
        }
    )
)


# =========================================================
# 5. GLOBAL ADJUSTED GENDER BOXPLOTS
# =========================================================

female_t2 = df.loc[
    df["Gender"] == 1,
    "T2_Adjusted_Gender"
]

male_t2 = df.loc[
    df["Gender"] == 0,
    "T2_Adjusted_Gender"
]


female_md = df.loc[
    df["Gender"] == 1,
    "MD_Adjusted_Gender"
]

male_md = df.loc[
    df["Gender"] == 0,
    "MD_Adjusted_Gender"
]


female_fa = df.loc[
    df["Gender"] == 1,
    "FA_Adjusted_Gender"
]

male_fa = df.loc[
    df["Gender"] == 0,
    "FA_Adjusted_Gender"
]


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})


fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.4)
)


boxprops = dict(
    edgecolor="black",
    linewidth=1.0
)

medianprops = dict(
    color="black",
    linewidth=1.2
)

whiskerprops = dict(
    color="black",
    linewidth=0.9
)

capprops = dict(
    color="black",
    linewidth=0.9
)

flierprops = dict(
    marker="o",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=0.7,
    markersize=3.5,
    linestyle="none"
)


def make_boxplot(
    ax,
    female_values,
    male_values,
    ylabel,
    p_value,
    panel_label
):

    bp = ax.boxplot(
        [
            female_values,
            male_values
        ],
        patch_artist=True,
        widths=0.48,
        showfliers=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops
    )

    # Female grey
    bp["boxes"][0].set_facecolor("0.75")

    # Male white
    bp["boxes"][1].set_facecolor("white")

    ax.set_xticks([1, 2])

    ax.set_xticklabels(
        [
            "Female",
            "Male"
        ]
    )

    ax.set_ylabel(
        ylabel
    )

    # Panel label
    ax.text(
        -0.13,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom"
    )

    # Regression p-value
    ax.text(
        0.50,
        0.92,
        rf"$p = {p_value:.3f}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.yaxis.grid(
        True,
        linewidth=0.4,
        alpha=0.15
    )

    ax.xaxis.grid(False)


make_boxplot(
    axes[0],
    female_t2,
    male_t2,
    r"Adjusted T$_2$ Relaxation Time (ms)",
    model_t2.pvalues["Gender"],
    "(a)"
)

make_boxplot(
    axes[1],
    female_md,
    male_md,
    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",
    model_md.pvalues["Gender"],
    "(b)"
)

make_boxplot(
    axes[2],
    female_fa,
    male_fa,
    "Adjusted Fractional Anisotropy",
    model_fa.pvalues["Gender"],
    "(c)"
)


plt.subplots_adjust(
    left=0.07,
    right=0.99,
    bottom=0.16,
    top=0.93,
    wspace=0.35
)


plt.savefig(
    "Adjusted_Global_qMRI_by_Gender_Boxplots.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.10
)

plt.show()


# =========================================================
# 6. REGIONAL ADJUSTED GENDER MODELS
# =========================================================
#
# Each model:
#
# Regional biomarker
# =
# Gender + Age + BMI
#
# Adjusted means are estimated with Age and BMI
# held at their cohort means.
# =========================================================

regions = [
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]

biomarkers = [
    "T2",
    "MD",
    "FA"
]


def get_adjusted_regional_results(biomarker):

    rows = []

    for region in regions:

        outcome = f"{biomarker}_{region}"

        model = smf.ols(
            f"{outcome} ~ Gender + Age + BMI",
            data=df
        ).fit()

        # Participant-level values adjusted for age and BMI.
        adjusted_values = (
            df[outcome]
            - model.params["Age"] * (df["Age"] - mean_age)
            - model.params["BMI"] * (df["BMI"] - mean_bmi)
        )

        female_values = adjusted_values[df["Gender"] == 1]
        male_values = adjusted_values[df["Gender"] == 0]

        # Adjusted means and 95% CIs used by the existing graph.
        prediction_data = pd.DataFrame({
            "Gender": [0, 1],
            "Age": [mean_age, mean_age],
            "BMI": [mean_bmi, mean_bmi]
        })

        prediction = model.get_prediction(
            prediction_data
        ).summary_frame(
            alpha=0.05
        )

        rows.append({
            "Biomarker": biomarker,
            "Region": region,
            "Male_mean": prediction.loc[0, "mean"],
            "Male_SD": male_values.std(ddof=1),
            "Male_lower": prediction.loc[0, "mean_ci_lower"],
            "Male_upper": prediction.loc[0, "mean_ci_upper"],
            "Female_mean": prediction.loc[1, "mean"],
            "Female_SD": female_values.std(ddof=1),
            "Female_lower": prediction.loc[1, "mean_ci_lower"],
            "Female_upper": prediction.loc[1, "mean_ci_upper"],
            "Adjusted_difference": model.params["Gender"],
            "p_value": model.pvalues["Gender"]
        })

    return pd.DataFrame(rows)


# =========================================================
# 7. CALCULATE REGIONAL RESULTS
# =========================================================

t2_regional = get_adjusted_regional_results(
    "T2"
)

md_regional = get_adjusted_regional_results(
    "MD"
)

fa_regional = get_adjusted_regional_results(
    "FA"
)


regional_results = pd.concat(
    [
        t2_regional,
        md_regional,
        fa_regional
    ],
    ignore_index=True
)


# =========================================================
# 8. REGIONAL RESULTS TABLE
# =========================================================

regional_table = regional_results[
    [
        "Biomarker",
        "Region",
        "Female_mean",
        "Female_SD",
        "Male_mean",
        "Male_SD",
        "Adjusted_difference",
        "p_value"
    ]
].copy()

regional_table.columns = [
    "Biomarker",
    "Region",
    "Female mean",
    "Female SD",
    "Male mean",
    "Male SD",
    "Female - Male",
    "p-value"
]

print("\nREGIONAL GENDER RESULTS")
print(
    regional_table.to_string(
        index=False,
        formatters={
            "Female mean": "{:.3f}".format,
            "Female SD": "{:.3f}".format,
            "Male mean": "{:.3f}".format,
            "Male SD": "{:.3f}".format,
            "Female - Male": "{:.3f}".format,
            "p-value": "{:.3f}".format
        }
    )
)


# =========================================================
# 9. REGIONAL ADJUSTED GENDER FIGURE
#
# Adjusted male and female means
# Error bars = 95% CI of adjusted mean
# p-value = Gender coefficient from regression
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.6)
)


x = np.arange(
    len(regions)
)

width = 0.34


def plot_regional_adjusted(
    ax,
    results,
    ylabel,
    panel_label
):

    male_mean = (
        results["Male_mean"].to_numpy()
    )

    female_mean = (
        results["Female_mean"].to_numpy()
    )


    # =====================================================
    # 95% CI ERROR BARS
    # =====================================================

    male_error = np.vstack([

        male_mean
        - results["Male_lower"].to_numpy(),

        results["Male_upper"].to_numpy()
        - male_mean
    ])


    female_error = np.vstack([

        female_mean
        - results["Female_lower"].to_numpy(),

        results["Female_upper"].to_numpy()
        - female_mean
    ])


    # =====================================================
    # MALE BARS
    # =====================================================

    ax.bar(
        x - width / 2,
        male_mean,
        width,

        label="Male",

        color="white",
        edgecolor="black",
        linewidth=0.9,

        yerr=male_error,

        error_kw=dict(
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8
        )
    )


    # =====================================================
    # FEMALE BARS
    # =====================================================

    ax.bar(
        x + width / 2,
        female_mean,
        width,

        label="Female",

        color="0.72",
        edgecolor="black",
        linewidth=0.9,

        yerr=female_error,

        error_kw=dict(
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8
        )
    )


    # =====================================================
    # X / Y LABELS
    # =====================================================

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regions
    )

    ax.set_xlabel(
        "Cartilage region"
    )

    ax.set_ylabel(
        ylabel
    )


    # =====================================================
    # PANEL LABEL
    # =====================================================

    ax.text(
        -0.10,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold"
    )


    # =====================================================
    # DETERMINE P-VALUE POSITION
    # =====================================================

    lowest = min(
        results["Male_lower"].min(),
        results["Female_lower"].min()
    )

    highest = max(
        results["Male_upper"].max(),
        results["Female_upper"].max()
    )

    data_range = (
        highest - lowest
    )


    # =====================================================
    # P-VALUES
    # =====================================================

    for i, row in results.reset_index(
        drop=True
    ).iterrows():

        p = row["p_value"]

        y_position = max(
            row["Male_upper"],
            row["Female_upper"]
        ) + data_range * 0.07


        if p < 0.001:

            p_text = (
                r"$p < 0.001$"
            )

        else:

            p_text = (
                rf"$p = {p:.3f}$"
            )


        ax.text(
            x[i],
            y_position,
            p_text,
            ha="center",
            va="bottom",
            fontsize=8.5
        )


    # =====================================================
    # AXIS LIMITS
    # =====================================================

    current_bottom = (
        ax.get_ylim()[0]
    )

    ax.set_ylim(
        bottom=current_bottom,
        top=highest + data_range * 0.28
    )


    # =====================================================
    # STYLE
    # =====================================================

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(0.8)

    ax.spines[
        "bottom"
    ].set_linewidth(0.8)


    ax.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
        length=3
    )


    ax.yaxis.grid(
        True,
        linewidth=0.4,
        alpha=0.15
    )

    ax.xaxis.grid(
        False
    )

    ax.set_axisbelow(
        True
    )


# =========================================================
# PANEL A — T2
# =========================================================

plot_regional_adjusted(
    axes[0],
    t2_regional,
    r"Adjusted T$_2$ Relaxation Time (ms)",
    "(a)"
)


# =========================================================
# PANEL B — MD
# =========================================================

plot_regional_adjusted(
    axes[1],
    md_regional,
    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",
    "(b)"
)


# =========================================================
# PANEL C — FA
# =========================================================

plot_regional_adjusted(
    axes[2],
    fa_regional,
    "Adjusted Fractional Anisotropy",
    "(c)"
)


# =========================================================
# SINGLE LEGEND
# =========================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.00),
    ncol=2,
    frameon=False,
    fontsize=9
)


# =========================================================
# FIGURE SPACING
# =========================================================

plt.subplots_adjust(
    left=0.07,
    right=0.99,
    bottom=0.16,
    top=0.87,
    wspace=0.32
)


# =========================================================
# SAVE REGIONAL FIGURE
# =========================================================

plt.savefig(
    "Adjusted_Regional_qMRI_by_Gender.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.10
)

plt.show()