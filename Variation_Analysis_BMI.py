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
# These are used when adjusting participant-level values
# and when generating model-predicted BMI trends.
# ============================================================

mean_gender = df["Gender"].mean()
mean_age = df["Age"].mean()


print("\n" + "=" * 65)
print("COHORT MEANS USED FOR BMI ADJUSTMENT")
print("=" * 65)

print(f"Gender mean       = {mean_gender:.3f}")
print(f"Age mean       = {mean_age:.3f}")


# ============================================================
# 4. DEFINE ALL GLOBAL AND REGIONAL OUTCOMES
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
# 5. FIT MULTIVARIABLE BMI MODELS
#
# qMRI outcome ~ BMI + Gender + Age
#
# BMI is the predictor of interest.
# ============================================================

models = {}


for outcome in outcomes:

    model = smf.ols(
        f"{outcome} ~ BMI + Gender + Age",
        data=df
    ).fit()

    models[outcome] = model


# ============================================================
# 6. CREATE PARTICIPANT-LEVEL VALUES ADJUSTED FOR
#    GENDER AND AGE
#
# BMI IS RETAINED.
#
# Y_adj =
#
# Y
# - beta_gender  * (Gender - mean Gender)
# - beta_age  * (Age - mean Age)
#
# This leaves:
#   BMI-related variation
#   residual participant variation
# ============================================================

def create_bmi_adjusted_values(outcome, model):

    return (

        df[outcome]

        - model.params["Gender"]
        * (df["Gender"] - mean_gender)

        - model.params["Age"]
        * (df["Age"] - mean_age)
    )


for outcome in outcomes:

    df[f"{outcome}_BMI_Adjusted"] = (
        create_bmi_adjusted_values(
            outcome,
            models[outcome]
        )
    )


# ============================================================
# 7. PRINT PARTICIPANT-LEVEL ADJUSTED T2 VALUES
# ============================================================

t2_display = df[[
    "Code",
    "BMI",
    "T2_Global_BMI_Adjusted",
    "T2_MFC_BMI_Adjusted",
    "T2_LFC_BMI_Adjusted",
    "T2_MTC_BMI_Adjusted",
    "T2_LTC_BMI_Adjusted"
]].copy()


t2_display.columns = [
    "Code",
    "BMI",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("BMI ANALYSIS: PARTICIPANT-LEVEL ADJUSTED T2 VALUES")
print("Adjusted for Gender and Age; BMI retained")
print("=" * 100)

print(
    t2_display.to_string(
        index=False,
        formatters={
            "BMI": "{:.2f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 8. PRINT PARTICIPANT-LEVEL ADJUSTED MD VALUES
# ============================================================

md_display = df[[
    "Code",
    "BMI",
    "MD_Global_BMI_Adjusted",
    "MD_MFC_BMI_Adjusted",
    "MD_LFC_BMI_Adjusted",
    "MD_MTC_BMI_Adjusted",
    "MD_LTC_BMI_Adjusted"
]].copy()


md_display.columns = [
    "Code",
    "BMI",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("BMI ANALYSIS: PARTICIPANT-LEVEL ADJUSTED MD VALUES")
print("Adjusted for Gender and Age; BMI retained")
print("=" * 100)

print(
    md_display.to_string(
        index=False,
        formatters={
            "BMI": "{:.2f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 9. PRINT PARTICIPANT-LEVEL ADJUSTED FA VALUES
# ============================================================

fa_display = df[[
    "Code",
    "BMI",
    "FA_Global_BMI_Adjusted",
    "FA_MFC_BMI_Adjusted",
    "FA_LFC_BMI_Adjusted",
    "FA_MTC_BMI_Adjusted",
    "FA_LTC_BMI_Adjusted"
]].copy()


fa_display.columns = [
    "Code",
    "BMI",
    "Global",
    "MFC",
    "LFC",
    "MTC",
    "LTC"
]


print("\n" + "=" * 100)
print("BMI ANALYSIS: PARTICIPANT-LEVEL ADJUSTED FA VALUES")
print("Adjusted for Gender and Age; BMI retained")
print("=" * 100)

print(
    fa_display.to_string(
        index=False,
        formatters={
            "BMI": "{:.2f}".format,
            "Global": "{:.3f}".format,
            "MFC": "{:.3f}".format,
            "LFC": "{:.3f}".format,
            "MTC": "{:.3f}".format,
            "LTC": "{:.3f}".format
        }
    )
)


# ============================================================
# 10. PRINT BMI REGRESSION RESULTS
#
# B = expected change in qMRI per 1 kg/m² increase in BMI,
# after adjustment for Gender and Age Sport.
# ============================================================

print("\n" + "=" * 100)
print("ADJUSTED BMI REGRESSION RESULTS")
print("=" * 100)


for outcome in outcomes:

    model = models[outcome]

    ci = model.conf_int().loc["BMI"]

    print(
        f"{outcome:12s} | "
        f"B = {model.params['BMI']:8.4f} | "
        f"95% CI = {ci.iloc[0]:8.4f} to {ci.iloc[1]:8.4f} | "
        f"p = {model.pvalues['BMI']:.4f}"
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
# 12. MODEL-BASED BMI PREDICTIONS + 95% CI
#
# This uses the ACTUAL multivariable model rather than
# fitting a second simple regression to the adjusted values.
#
# Gender and Age are held at cohort means.
# ============================================================

def get_bmi_prediction(model, x_pred):

    prediction_data = pd.DataFrame({

        "BMI":
            x_pred,

        "Gender":
            np.repeat(
                mean_gender,
                len(x_pred)
            ),

        "Age":
            np.repeat(
                mean_age,
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
# 13. GLOBAL BMI PANEL
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
    # BMI p-value from multivariable OLS
    # --------------------------------------------------------

    p = (
        model.pvalues["BMI"]
    )


    # --------------------------------------------------------
    # BMI range
    # --------------------------------------------------------

    x_pred = np.linspace(
        np.floor(np.min(x)),
        np.ceil(np.max(x)),
        6
    )


    # --------------------------------------------------------
    # Predicted adjusted means + 95% CI from full model
    # --------------------------------------------------------

    y_pred, lower, upper = (
        get_bmi_prediction(
            model,
            x_pred
        )
    )


    yerr = np.vstack([
        y_pred - lower,
        upper - y_pred
    ])


    # --------------------------------------------------------
    # OPTIONAL INDIVIDUAL ADJUSTED VALUES
    #
    # These show the 30 participants.
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
    # MODEL-BASED CONNECTING LINE
    # --------------------------------------------------------

    ax.plot(
        x_pred,
        y_pred,
        color="0.25",
        linewidth=1.3,
        zorder=2
    )


    # --------------------------------------------------------
    # MODEL-BASED MEANS + 95% CI
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
    # AXES
    # --------------------------------------------------------

    ax.set_xlabel(
        r"BMI (kg/m$^2$)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlim(
        np.floor(np.min(x)) - 1,
        np.ceil(np.max(x)) + 1
    )

    style_axis(ax)


    # --------------------------------------------------------
    # ADJUSTED P-VALUE ONLY
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
    # PANEL LETTER
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
# 14. GLOBAL BMI FIGURE
# ============================================================

BMI = df["BMI"].to_numpy()


fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.8, 4.2)
)


print("\n" + "=" * 70)
print("GLOBAL ADJUSTED BMI ASSOCIATIONS")
print("=" * 70)


# ------------------------------------------------------------
# T2
# ------------------------------------------------------------

p = global_panel(
    axes[0],

    BMI,

    df[
        "T2_Global_BMI_Adjusted"
    ].to_numpy(),

    models[
        "T2_Global"
    ],

    r"Adjusted T$_2$ Relaxation Time (ms)",

    "a"
)

print(
    f"T2: adjusted BMI p = {p:.4f}"
)


# ------------------------------------------------------------
# MD
# ------------------------------------------------------------

p = global_panel(
    axes[1],

    BMI,

    df[
        "MD_Global_BMI_Adjusted"
    ].to_numpy(),

    models[
        "MD_Global"
    ],

    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",

    "b"
)

print(
    f"MD: adjusted BMI p = {p:.4f}"
)


# ------------------------------------------------------------
# FA
# ------------------------------------------------------------

p = global_panel(
    axes[2],

    BMI,

    df[
        "FA_Global_BMI_Adjusted"
    ].to_numpy(),

    models[
        "FA_Global"
    ],

    "Adjusted Fractional Anisotropy",

    "c"
)

print(
    f"FA: adjusted BMI p = {p:.4f}"
)


plt.tight_layout(
    w_pad=2.2
)


plt.savefig(
    "BMI_Global_Adjusted_qMRI.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)


plt.show()


# ============================================================
# 15. REGIONAL ADJUSTED DATA
# ============================================================

regional_T2 = {

    "MFC":
        df["T2_MFC_BMI_Adjusted"].to_numpy(),

    "LFC":
        df["T2_LFC_BMI_Adjusted"].to_numpy(),

    "MTC":
        df["T2_MTC_BMI_Adjusted"].to_numpy(),

    "LTC":
        df["T2_LTC_BMI_Adjusted"].to_numpy()
}


regional_MD = {

    "MFC":
        df["MD_MFC_BMI_Adjusted"].to_numpy(),

    "LFC":
        df["MD_LFC_BMI_Adjusted"].to_numpy(),

    "MTC":
        df["MD_MTC_BMI_Adjusted"].to_numpy(),

    "LTC":
        df["MD_LTC_BMI_Adjusted"].to_numpy()
}


regional_FA = {

    "MFC":
        df["FA_MFC_BMI_Adjusted"].to_numpy(),

    "LFC":
        df["FA_LFC_BMI_Adjusted"].to_numpy(),

    "MTC":
        df["FA_MTC_BMI_Adjusted"].to_numpy(),

    "LTC":
        df["FA_LTC_BMI_Adjusted"].to_numpy()
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
# 18. REGIONAL BMI PANEL
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
        np.floor(np.min(x)),
        np.ceil(np.max(x)),
        6
    )


    results = {}


    for region, adjusted_y in regional_values.items():

        model = (
            regional_models[region]
        )

        p = (
            model.pvalues["BMI"]
        )

        results[region] = p


        # ----------------------------------------------------
        # MODEL-BASED PREDICTION
        # ----------------------------------------------------

        y_pred, lower, upper = (
            get_bmi_prediction(
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
        # INDIVIDUAL PARTICIPANT ADJUSTED VALUES
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
        # MODEL-BASED TREND
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
        # MODEL-BASED MEAN + 95% CI
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
    # AXES
    # --------------------------------------------------------

    ax.set_xlabel(
        r"BMI (kg/m$^2$)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlim(
        np.floor(np.min(x)) - 1,
        np.ceil(np.max(x)) + 1
    )

    style_axis(ax)


    # --------------------------------------------------------
    # PANEL LETTER
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
    # LEGEND
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
# 19. REGIONAL BMI FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.7)
)


print("\n" + "=" * 70)
print("REGIONAL ADJUSTED BMI ASSOCIATIONS")
print("=" * 70)


# ------------------------------------------------------------
# T2
# ------------------------------------------------------------

results_T2 = regional_panel(
    axes[0],

    BMI,

    regional_T2,

    regional_T2_models,

    r"Adjusted T$_2$ Relaxation Time (ms)",

    "a"
)


print("\nT2")

for region, p in results_T2.items():

    print(
        f"{region}: "
        f"adjusted BMI p = {p:.4f}"
    )


# ------------------------------------------------------------
# MD
# ------------------------------------------------------------

results_MD = regional_panel(
    axes[1],

    BMI,

    regional_MD,

    regional_MD_models,

    r"Adjusted MD ($\times 10^{-3}$ mm$^2$/s)",

    "b"
)


print("\nMD")

for region, p in results_MD.items():

    print(
        f"{region}: "
        f"adjusted BMI p = {p:.4f}"
    )


# ------------------------------------------------------------
# FA
# ------------------------------------------------------------

results_FA = regional_panel(
    axes[2],

    BMI,

    regional_FA,

    regional_FA_models,

    "Adjusted Fractional Anisotropy",

    "c"
)


print("\nFA")

for region, p in results_FA.items():

    print(
        f"{region}: "
        f"adjusted BMI p = {p:.4f}"
    )


plt.tight_layout(
    w_pad=2.4
)


plt.savefig(
    "BMI_Regional_Adjusted_qMRI.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)


plt.show()