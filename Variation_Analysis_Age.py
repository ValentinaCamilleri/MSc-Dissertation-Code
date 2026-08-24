import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# ============================================================
# 1. LOAD PARTICIPANT DATA FROM EXCEL
# ============================================================

EXCEL_FILE = r"C:\Users\Valentina\Documents\Masters_Data.xlsx"

df = pd.read_excel(EXCEL_FILE)

# Convert headings such as "T2 MFC" to "T2_MFC".
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

# Excel stores Sex as M/F.
# Convert to the coding used by this analysis:
# Male = 0, Female = 1
df["Gender"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"M": 0, "F": 1})
)


# ============================================================
# 3. COHORT MEANS
#
# Age is NOT centred out because it is the predictor
# of interest.
# ============================================================

mean_gender = df["Gender"].mean()
mean_bmi = df["BMI"].mean()


print("\n" + "=" * 65)
print("COHORT MEANS USED FOR AGE ADJUSTMENT")
print("=" * 65)

print(f"Gender mean = {mean_gender:.3f}")
print(f"BMI mean    = {mean_bmi:.3f}")


# ============================================================
# 4. DEFINE ALL OUTCOMES
# ============================================================

outcomes = [

    # T2
    "T2_Global",
    "T2_MFC",
    "T2_LFC",
    "T2_MTC",
    "T2_LTC",

    # MD
    "MD_Global",
    "MD_MFC",
    "MD_LFC",
    "MD_MTC",
    "MD_LTC",

    # FA
    "FA_Global",
    "FA_MFC",
    "FA_LFC",
    "FA_MTC",
    "FA_LTC"
]


# ============================================================
# 5. FIT MULTIVARIABLE AGE MODELS
#
# qMRI outcome ~ Age + Gender + BMI
#
# Age is the predictor of interest.
# ============================================================

models = {}


for outcome in outcomes:

    model = smf.ols(
        f"{outcome} ~ Age + Gender + BMI",
        data=df
    ).fit()

    models[outcome] = model


# ============================================================
# 6. CREATE PARTICIPANT-LEVEL VALUES ADJUSTED FOR
#    GENDER AND BMI
#
# AGE IS RETAINED.
#
# Y_adj =
#
# Y
# - beta_gender  * (Gender - mean Gender)
# - beta_BMI  * (BMI - mean BMI)
#
# This leaves:
#   age-related variation
#   residual participant variation
# ============================================================

def create_age_adjusted_values(outcome, model):

    return (

        df[outcome]

        - model.params["Gender"]
        * (df["Gender"] - mean_gender)

        - model.params["BMI"]
        * (df["BMI"] - mean_bmi)
    )


for outcome in outcomes:

    df[f"{outcome}_Age_Adjusted"] = (
        create_age_adjusted_values(
            outcome,
            models[outcome]
        )
    )


# ============================================================
# 7. PRINT PARTICIPANT-LEVEL AGE-ADJUSTED T2 VALUES
# ============================================================

t2_display = df[[
    "Code",
    "Age",
    "T2_Global_Age_Adjusted",
    "T2_MFC_Age_Adjusted",
    "T2_LFC_Age_Adjusted",
    "T2_MTC_Age_Adjusted",
    "T2_LTC_Age_Adjusted"
]].copy()


t2_display.columns = [
    "Code",
    "Age",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("AGE ANALYSIS: PARTICIPANT-LEVEL ADJUSTED T2 VALUES")
print("Adjusted for Gender and BMI; Age retained")
print("=" * 100)

print(
    t2_display.to_string(
        index=False,
        formatters={
            "Age": "{:.0f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 8. PRINT PARTICIPANT-LEVEL AGE-ADJUSTED MD VALUES
# ============================================================

md_display = df[[
    "Code",
    "Age",
    "MD_Global_Age_Adjusted",
    "MD_MFC_Age_Adjusted",
    "MD_LFC_Age_Adjusted",
    "MD_MTC_Age_Adjusted",
    "MD_LTC_Age_Adjusted"
]].copy()


md_display.columns = [
    "Code",
    "Age",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("AGE ANALYSIS: PARTICIPANT-LEVEL ADJUSTED MD VALUES")
print("Adjusted for Gender and BMI; Age retained")
print("=" * 100)

print(
    md_display.to_string(
        index=False,
        formatters={
            "Age": "{:.0f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 9. PRINT PARTICIPANT-LEVEL AGE-ADJUSTED FA VALUES
# ============================================================

fa_display = df[[
    "Code",
    "Age",
    "FA_Global_Age_Adjusted",
    "FA_MFC_Age_Adjusted",
    "FA_LFC_Age_Adjusted",
    "FA_MTC_Age_Adjusted",
    "FA_LTC_Age_Adjusted"
]].copy()


fa_display.columns = [
    "Code",
    "Age",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("AGE ANALYSIS: PARTICIPANT-LEVEL ADJUSTED FA VALUES")
print("Adjusted for Gender and BMI; Age retained")
print("=" * 100)

print(
    fa_display.to_string(
        index=False,
        formatters={
            "Age": "{:.0f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 10. PRINT AGE REGRESSION RESULTS
#
# B = expected qMRI change per 1-year increase in age,
# adjusted for Gender and BMI.
# ============================================================

print("\n" + "=" * 100)
print("ADJUSTED AGE REGRESSION RESULTS")
print("=" * 100)


for outcome in outcomes:

    model = models[outcome]

    ci = model.conf_int().loc["Age"]

    print(
        f"{outcome:12s} | "
        f"B = {model.params['Age']:8.4f} | "
        f"95% CI = {ci.iloc[0]:8.4f} to {ci.iloc[1]:8.4f} | "
        f"p = {model.pvalues['Age']:.4f}"
    )


# ============================================================
# 11. FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8
})


def format_p(p):

    if p < 0.001:
        return r"$p < 0.001$"

    return rf"$p = {p:.3f}$"


def style_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        width=0.8,
        length=3
    )

    ax.yaxis.grid(
        True,
        linewidth=0.4,
        alpha=0.15
    )

    ax.xaxis.grid(False)

    ax.set_axisbelow(True)


# ============================================================
# 12. MODEL-BASED AGE PREDICTIONS + 95% CI
#
# Predictions come from the full multivariable OLS model.
#
# Gender and BMI are held at cohort means.
# ============================================================

def get_age_prediction(model, x_pred):

    prediction_data = pd.DataFrame({

        "Age":
            x_pred,

        "Gender":
            np.repeat(
                mean_gender,
                len(x_pred)
            ),

        "BMI":
            np.repeat(
                mean_bmi,
                len(x_pred)
            )
    })


    prediction = model.get_prediction(
        prediction_data
    ).summary_frame(
        alpha=0.05
    )


    y_pred = (
        prediction["mean"]
        .to_numpy()
    )

    lower = (
        prediction["mean_ci_lower"]
        .to_numpy()
    )

    upper = (
        prediction["mean_ci_upper"]
        .to_numpy()
    )


    return (
        y_pred,
        lower,
        upper
    )


# ============================================================
# 13. GLOBAL AGE PANEL
# ============================================================

def global_panel(
    ax,
    x,
    adjusted_y,
    model,
    ylabel,
    letter
):

    # --------------------------------------------------------
    # Adjusted Age p-value
    # --------------------------------------------------------

    p = (
        model.pvalues["Age"]
    )


    # --------------------------------------------------------
    # Age prediction range
    # --------------------------------------------------------

    x_pred = np.linspace(
        np.min(x),
        np.max(x),
        6
    )


    # --------------------------------------------------------
    # Model-based predicted means + 95% CI
    # --------------------------------------------------------

    y_pred, lower, upper = (
        get_age_prediction(
            model,
            x_pred
        )
    )


    yerr = np.vstack([
        y_pred - lower,
        upper - y_pred
    ])


    # --------------------------------------------------------
    # Individual participant adjusted values
    # --------------------------------------------------------

    ax.scatter(
        x,
        adjusted_y,

        s=18,

        facecolors="white",
        edgecolors="0.45",

        linewidths=0.7,
        alpha=0.65,

        zorder=1
    )


    # --------------------------------------------------------
    # Model-based trend line
    # --------------------------------------------------------

    ax.plot(
        x_pred,
        y_pred,

        color="0.25",
        linewidth=1.3,

        zorder=2
    )


    # --------------------------------------------------------
    # Model-based means + 95% CI
    # --------------------------------------------------------

    ax.errorbar(
        x_pred,
        y_pred,

        yerr=yerr,

        fmt="o",
        markersize=4.0,

        color="0.20",
        ecolor="0.50",

        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,

        markerfacecolor="0.20",
        markeredgewidth=0,

        zorder=3
    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel(
        "Age (years)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlim(
        np.min(x) - 1,
        np.max(x) + 1
    )

    style_axis(ax)


    # --------------------------------------------------------
    # Adjusted p-value
    # --------------------------------------------------------

    ax.text(
        0.97,
        0.96,

        format_p(p),

        transform=ax.transAxes,

        ha="right",
        va="top",

        fontsize=10
    )


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.15,
        1.03,

        f"({letter})",

        transform=ax.transAxes,

        fontsize=11,
        fontweight="bold"
    )


    return p


# ============================================================
# 14. GLOBAL AGE FIGURE
# ============================================================

AGE = df["Age"].to_numpy()


fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.8, 4.2)
)


print("\n" + "=" * 70)
print("GLOBAL ADJUSTED AGE ASSOCIATIONS")
print("=" * 70)


# ------------------------------------------------------------
# T2
# ------------------------------------------------------------

p = global_panel(
    axes[0],

    AGE,

    df[
        "T2_Global_Age_Adjusted"
    ].to_numpy(),

    models[
        "T2_Global"
    ],

    r"Adjusted T$_2$ Relaxation Time (ms)",

    "a"
)


print(
    f"T2: adjusted Age p = {p:.4f}"
)


# ------------------------------------------------------------
# MD
# ------------------------------------------------------------

p = global_panel(
    axes[1],

    AGE,

    df[
        "MD_Global_Age_Adjusted"
    ].to_numpy(),

    models[
        "MD_Global"
    ],

    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",

    "b"
)


print(
    f"MD: adjusted Age p = {p:.4f}"
)


# ------------------------------------------------------------
# FA
# ------------------------------------------------------------

p = global_panel(
    axes[2],

    AGE,

    df[
        "FA_Global_Age_Adjusted"
    ].to_numpy(),

    models[
        "FA_Global"
    ],

    "Adjusted Fractional Anisotropy",

    "c"
)


print(
    f"FA: adjusted Age p = {p:.4f}"
)


plt.tight_layout(
    w_pad=2.2
)


plt.savefig(
    "Age_Global_Adjusted_qMRI.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)


plt.show()


# ============================================================
# 15. REGIONAL AGE-ADJUSTED DATA
# ============================================================

regional_T2 = {

    "MFC":
        df["T2_MFC_Age_Adjusted"].to_numpy(),

    "LFC":
        df["T2_LFC_Age_Adjusted"].to_numpy(),

    "MTC":
        df["T2_MTC_Age_Adjusted"].to_numpy(),

    "LTC":
        df["T2_LTC_Age_Adjusted"].to_numpy()
}


regional_MD = {

    "MFC":
        df["MD_MFC_Age_Adjusted"].to_numpy(),

    "LFC":
        df["MD_LFC_Age_Adjusted"].to_numpy(),

    "MTC":
        df["MD_MTC_Age_Adjusted"].to_numpy(),

    "LTC":
        df["MD_LTC_Age_Adjusted"].to_numpy()
}


regional_FA = {

    "MFC":
        df["FA_MFC_Age_Adjusted"].to_numpy(),

    "LFC":
        df["FA_LFC_Age_Adjusted"].to_numpy(),

    "MTC":
        df["FA_MTC_Age_Adjusted"].to_numpy(),

    "LTC":
        df["FA_LTC_Age_Adjusted"].to_numpy()
}


# ============================================================
# 16. REGIONAL MODELS
# ============================================================

regional_T2_models = {
    region: models[f"T2_{region}"]
    for region in [
        "MFC",
        "LFC",
        "MTC",
        "LTC"
    ]
}


regional_MD_models = {
    region: models[f"MD_{region}"]
    for region in [
        "MFC",
        "LFC",
        "MTC",
        "LTC"
    ]
}


regional_FA_models = {
    region: models[f"FA_{region}"]
    for region in [
        "MFC",
        "LFC",
        "MTC",
        "LTC"
    ]
}


# ============================================================
# 17. REGIONAL LINE STYLES
# ============================================================

region_styles = {

    "MFC": {
        "color": "0.15",
        "linestyle": "-",
        "marker": "o"
    },

    "LFC": {
        "color": "0.35",
        "linestyle": "--",
        "marker": "s"
    },

    "MTC": {
        "color": "0.52",
        "linestyle": "-.",
        "marker": "^"
    },

    "LTC": {
        "color": "0.68",
        "linestyle": ":",
        "marker": "D"
    }
}


# ============================================================
# 18. REGIONAL AGE PANEL
# ============================================================

def regional_panel(
    ax,
    x,
    regional_values,
    regional_models,
    ylabel,
    letter
):

    x_pred = np.linspace(
        np.min(x),
        np.max(x),
        6
    )


    results = {}


    for region, adjusted_y in regional_values.items():

        model = (
            regional_models[region]
        )


        # ----------------------------------------------------
        # Adjusted Age p-value
        # ----------------------------------------------------

        p = (
            model.pvalues["Age"]
        )

        results[region] = p


        # ----------------------------------------------------
        # Model-based predictions
        # ----------------------------------------------------

        y_pred, lower, upper = (
            get_age_prediction(
                model,
                x_pred
            )
        )


        yerr = np.vstack([
            y_pred - lower,
            upper - y_pred
        ])


        style = (
            region_styles[region]
        )


        # ----------------------------------------------------
        # Individual participant adjusted values
        # ----------------------------------------------------

        ax.scatter(
            x,
            adjusted_y,

            s=10,

            facecolors="none",
            edgecolors=style["color"],

            linewidths=0.45,
            alpha=0.18,

            zorder=1
        )


        # ----------------------------------------------------
        # Model-based trend
        # ----------------------------------------------------

        ax.plot(
            x_pred,
            y_pred,

            color=style["color"],
            linestyle=style["linestyle"],

            linewidth=1.25,

            zorder=2
        )


        # ----------------------------------------------------
        # Predicted means + 95% CI
        # ----------------------------------------------------

        ax.errorbar(
            x_pred,
            y_pred,

            yerr=yerr,

            fmt=style["marker"],
            markersize=3.4,

            color=style["color"],
            ecolor=style["color"],

            elinewidth=0.8,
            capsize=2,

            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.8,

            alpha=0.95,

            label=(
                f"{region}: "
                + format_p(p)
            ),

            zorder=3
        )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel(
        "Age (years)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlim(
        np.min(x) - 1,
        np.max(x) + 1
    )

    style_axis(ax)


    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        -0.15,
        1.03,

        f"({letter})",

        transform=ax.transAxes,

        fontsize=11,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="best",
        frameon=False,
        fontsize=7.2,
        handlelength=2.3,
        labelspacing=0.45
    )


    return results


# ============================================================
# 19. REGIONAL AGE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.7)
)


print("\n" + "=" * 70)
print("REGIONAL ADJUSTED AGE ASSOCIATIONS")
print("=" * 70)


# ------------------------------------------------------------
# T2
# ------------------------------------------------------------

results_T2 = regional_panel(
    axes[0],

    AGE,

    regional_T2,

    regional_T2_models,

    r"Adjusted T$_2$ Relaxation Time (ms)",

    "a"
)


print("\nT2")

for region, p in results_T2.items():

    print(
        f"{region}: "
        f"adjusted Age p = {p:.4f}"
    )


# ------------------------------------------------------------
# MD
# ------------------------------------------------------------

results_MD = regional_panel(
    axes[1],

    AGE,

    regional_MD,

    regional_MD_models,

    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",

    "b"
)


print("\nMD")

for region, p in results_MD.items():

    print(
        f"{region}: "
        f"adjusted Age p = {p:.4f}"
    )


# ------------------------------------------------------------
# FA
# ------------------------------------------------------------

results_FA = regional_panel(
    axes[2],

    AGE,

    regional_FA,

    regional_FA_models,

    "Adjusted Fractional Anisotropy",

    "c"
)


print("\nFA")

for region, p in results_FA.items():

    print(
        f"{region}: "
        f"adjusted Age p = {p:.4f}"
    )


plt.tight_layout(
    w_pad=2.4
)


plt.savefig(
    "Age_Regional_Adjusted_qMRI.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)


plt.show()