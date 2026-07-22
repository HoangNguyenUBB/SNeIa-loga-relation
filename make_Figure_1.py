"""
Create Figure 1.

Required input files
--------------------
Figure_1_D1_original.txt
Figure_1_D2_age_corrected.txt

These files are produced by make_Table_D1_and_D2.py and contain:

    z_mean
    Delta_mu
    sigma_Delta_mu

Output
------
Figure_1.pdf
Figure_1.png

------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# Compatibility across SciPy versions:
# SciPy >= 1.10 uses cumulative_trapezoid; older versions use cumtrapz.
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# ============================================================
# Constants
# ============================================================

C_LIGHT = 299792.458  # km/s

# Fiducial Einstein-de Sitter reference
H0_EDS = 73.00

# Original Pantheon+ flat Lambda-CDM fit
H0_LCDM_ORIGINAL = 72.84
OMEGA_LAMBDA_ORIGINAL = 0.638

# Age-corrected Pantheon+ flat Lambda-CDM fit
H0_LCDM_CORRECTED = 73.06
OMEGA_LAMBDA_CORRECTED = 0.462

# Age-corrected Pantheon+ Loga fit
H0_LOGA_CORRECTED = 72.56


# ============================================================
# Input files
# ============================================================

ORIGINAL_FILE = "Figure_1_D1_original.txt"
CORRECTED_FILE = "Figure_1_D2_age_corrected.txt"


# ============================================================
# Read binned Pantheon+ outputs
# ============================================================

original = np.genfromtxt(
    ORIGINAL_FILE,
    names=True,
    encoding="utf-8",
)

corrected = np.genfromtxt(
    CORRECTED_FILE,
    names=True,
    encoding="utf-8",
)


z_original = original["z_mean"]
delta_mu_original = original["Delta_mu"]
sigma_original = original["sigma_Delta_mu"]

z_corrected = corrected["z_mean"]
delta_mu_corrected = corrected["Delta_mu"]
sigma_corrected = corrected["sigma_Delta_mu"]


# ============================================================
# Distance-modulus models
# ============================================================

def distance_modulus(d_l):
    """
    Convert luminosity distance in Mpc to distance modulus.
    """

    return 5.0 * np.log10(d_l) + 25.0


def mu_eds(z, h0):
    """
    Einstein-de Sitter luminosity distance:

        d_L = (2c/H0)(1+z)[1 - 1/sqrt(1+z)].
    """

    d_l = (
        2.0
        * C_LIGHT
        / h0
        * (1.0 + z)
        * (1.0 - 1.0 / np.sqrt(1.0 + z))
    )

    return distance_modulus(d_l)


def mu_lcdm(z, h0, omega_lambda):
    """
    Flat Lambda-CDM distance modulus.

    The input redshift array must be sorted and strictly positive.
    """

    integration_grid = np.concatenate((
        [0.0],
        np.asarray(z),
    ))

    omega_m = 1.0 - omega_lambda

    e_grid = np.sqrt(
        omega_m * (1.0 + integration_grid)**3
        + omega_lambda
    )

    integral_grid = cumtrapz(
        1.0 / e_grid,
        integration_grid,
        initial=0.0,
    )

    integral = integral_grid[1:]

    d_l = (
        C_LIGHT
        / h0
        * (1.0 + z)
        * integral
    )

    return distance_modulus(d_l)


def mu_loga(z, h0):
    """
    Logarithmic luminosity-distance relation:

        d_L = (c/H0)(1+z) ln(1+z).
    """

    d_l = (
        C_LIGHT
        / h0
        * (1.0 + z)
        * np.log(1.0 + z)
    )

    return distance_modulus(d_l)


# ============================================================
# Smooth model curves
# ============================================================

z_curve = np.linspace(
    1.0e-4,
    2.30,
    4000,
)

mu_eds_curve = mu_eds(
    z_curve,
    H0_EDS,
)


# Original flat Lambda-CDM relative to EdS

delta_mu_lcdm_original = (
    mu_lcdm(
        z_curve,
        H0_LCDM_ORIGINAL,
        OMEGA_LAMBDA_ORIGINAL,
    )
    - mu_eds_curve
)


# Age-corrected flat Lambda-CDM relative to EdS

delta_mu_lcdm_corrected = (
    mu_lcdm(
        z_curve,
        H0_LCDM_CORRECTED,
        OMEGA_LAMBDA_CORRECTED,
    )
    - mu_eds_curve
)


# Age-corrected Loga relative to EdS

delta_mu_loga_corrected = (
    mu_loga(
        z_curve,
        H0_LOGA_CORRECTED,
    )
    - mu_eds_curve
)


# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


fig, ax = plt.subplots(
    figsize=(5.4, 3.65),
)


# EdS reference

ax.axhline(
    0.0,
    color="tab:blue",
    linestyle=":",
    linewidth=1.2,
    label="EdS fiducial",
)


# Original Lambda-CDM

ax.plot(
    z_curve,
    delta_mu_lcdm_original,
    color="red",
    linestyle="--",
    linewidth=1.4,
    label=r"Flat $\Lambda$CDM, original",
)


# Age-corrected Lambda-CDM

ax.plot(
    z_curve,
    delta_mu_lcdm_corrected,
    color="green",
    linestyle="-.",
    linewidth=1.4,
    label=r"Flat $\Lambda$CDM, age-corrected",
)


# Loga relation

ax.plot(
    z_curve,
    delta_mu_loga_corrected,
    color="black",
    linestyle="-",
    linewidth=1.4,
    label=r"$d_L=\frac{c}{H_0}(1+z)\,\ln(1+z)$ relation",
)


# Original Pantheon+ bins

ax.errorbar(
    z_original,
    delta_mu_original,
    yerr=sigma_original,
    fmt="o",
    color="red",
    markersize=4.0,
    markeredgewidth=0.4,
    elinewidth=0.8,
    capsize=1.7,
    label="Original Pan+",
    zorder=5,
)


# Age-corrected Pantheon+ bins

ax.errorbar(
    z_corrected,
    delta_mu_corrected,
    yerr=sigma_corrected,
    fmt="s",
    color="green",
    markersize=4.0,
    markeredgewidth=0.4,
    elinewidth=0.8,
    capsize=1.7,
    label="Age-corrected Pan+",
    zorder=6,
)


# ============================================================
# Formatting
# ============================================================

ax.set_xlim(
    0,
    2.30,
)

ax.set_ylim(
    0,
    1.00,
)

ax.set_xlabel(
    r"Redshift $z$"
)

ax.set_ylabel(
    r"$\Delta\mu$ relative to EdS [mag]"
)

ax.set_xticks([
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
])

ax.set_yticks(
    np.arange(
        0.0,
        1.01,
        0.2,
    )
)

ax.grid(
    True,
    linestyle="-",
    linewidth=0.4,
    alpha=0.20,
)

ax.tick_params(
    direction="in",
    top=True,
    right=True,
)

ax.minorticks_on()

ax.tick_params(
    which="minor",
    direction="in",
    top=True,
    right=True,
)

ax.legend(
    loc="upper left",
    frameon=False,
    handlelength=2.4,
    borderaxespad=0.5,
)

fig.tight_layout(
    pad=0.5,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    "Figure_1.pdf",
    bbox_inches="tight",
)

fig.savefig(
    "Figure_1.png",
    dpi=400,
    bbox_inches="tight",
)

print()
print("Saved Figure_1.pdf")
print("Saved Figure_1.png")
