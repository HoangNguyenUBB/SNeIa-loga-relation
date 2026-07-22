"""
Create Tables D.1 and D.2 from Pantheon+ only.

Table D.1
---------
Original Pantheon+ distance-modulus offsets relative to a
fiducial Einstein-de Sitter model with H0_EdS = 73.00.

The flat Lambda-CDM model is fitted to the original Pantheon+
distance moduli.

Table D.2
---------
Progenitor-age-corrected Pantheon+ distance-modulus offsets
relative to the same fiducial Einstein-de Sitter model.

Flat Lambda-CDM and Loga are fitted to the age-corrected
Pantheon+ distance moduli.

Age correction
--------------
    Delta_mu_age(z) = 0.183 [1 - exp(-2.2 z)]

Redshift bins
-------------
    [0.0, 0.1)
    [0.1, 0.2)
    [0.2, 0.3)
    [0.3, 0.4)
    [0.4, 0.5)
    [0.5, 0.6)
    [0.6, 0.7)
    [0.7, 1.0)
    [1.0, 2.3]

Outputs
-------
Table_D1.txt
Table_D2.txt
Figure_1_D1_original.txt
Figure_1_D2_age_corrected.txt

------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import pandas as pd

# Compatibility across SciPy versions:
# SciPy >= 1.10 uses cumulative_trapezoid; older versions use cumtrapz.
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# Constants
# ============================================================

C_LIGHT = 299792.458

AGE_DELTA = 0.183
AGE_KAPPA = 2.2

H0_EDS = 73.00


# ============================================================
# Fixed redshift bins
# ============================================================

BIN_EDGES = np.array([
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    1.0,
    2.3,
])

N_BINS = len(BIN_EDGES) - 1


# ============================================================
# Read Pantheon+
# ============================================================

df = pd.read_csv(
    "Pantheon+SH0ES.dat",
    sep=r"\s+",
    comment="#",
    encoding="utf-8",
)

z = df["zHD"].values
zHEL = df["zHEL"].values
mu_original = df["MU_SH0ES"].values

cov = np.load(
    "Pantheon+SH0ES_STAT+SYS.npy"
)

assert cov.shape == (
    len(z),
    len(z),
)

N = len(z)


# ============================================================
# Age-corrected Pantheon+
# ============================================================

age_correction = AGE_DELTA * (
    1.0 - np.exp(-AGE_KAPPA * z)
)

mu_corrected = (
    mu_original
    - age_correction
)


# ============================================================
# Full-covariance whitening
# ============================================================

L = cholesky(
    cov,
    lower=True,
    check_finite=False,
)


# ============================================================
# Integration grid
# ============================================================

z_grid, inverse_index = np.unique(
    np.concatenate(([0.0], z)),
    return_inverse=True,
)

z_to_grid = inverse_index[1:]


# ============================================================
# Distance-modulus helper
# ============================================================

def distance_modulus(d_l):

    return 5.0 * np.log10(d_l) + 25.0


# ============================================================
# Fiducial Einstein-de Sitter model
# ============================================================

def mu_eds_data():
    """
    EdS distance modulus at the individual Pantheon+ redshifts.
    H0 is fixed at H0_EDS.
    """

    shape = 2.0 * (
        1.0
        - 1.0 / np.sqrt(1.0 + z)
    )

    d_l = (
        (C_LIGHT / H0_EDS)
        * (1.0 + zHEL)
        * shape
    )

    return distance_modulus(d_l)


def mu_eds_at_redshift(z_value):
    """
    EdS distance modulus for a theoretical curve evaluated at
    one redshift, with z_hel identified with z.
    """

    shape = 2.0 * (
        1.0
        - 1.0 / np.sqrt(1.0 + z_value)
    )

    d_l = (
        (C_LIGHT / H0_EDS)
        * (1.0 + z_value)
        * shape
    )

    return distance_modulus(d_l)


# ============================================================
# Flat Lambda-CDM model
# ============================================================

def lcdm_integral(omega_lambda):

    e_squared = (
        (1.0 - omega_lambda)
        * (1.0 + z_grid)**3
        + omega_lambda
    )

    integral_grid = cumtrapz(
        1.0 / np.sqrt(e_squared),
        z_grid,
        initial=0.0,
    )

    return integral_grid[z_to_grid]


def mu_lcdm(params):
    """
    params[0] = H0
    params[1] = Omega_Lambda
    """

    h0, omega_lambda = params

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + zHEL)
        * lcdm_integral(omega_lambda)
    )

    return distance_modulus(d_l)


def fit_lcdm(mu_data):

    def residuals(params):

        return solve_triangular(
            L,
            mu_data - mu_lcdm(params),
            lower=True,
            check_finite=False,
        )

    result = least_squares(
        residuals,
        x0=[
            73.0,
            0.65,
        ],
        bounds=(
            [
                30.0,
                0.0,
            ],
            [
                120.0,
                1.0,
            ],
        ),
        method="trf",
        jac="2-point",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=10000,
    )

    if not result.success:
        raise RuntimeError(
            f"LCDM fit failed: {result.message}"
        )

    h0, omega_lambda = result.x

    parameter_covariance = np.linalg.inv(
        result.jac.T @ result.jac
    )

    errors = np.sqrt(
        np.diag(parameter_covariance)
    )

    return {
        "H0": h0,
        "sigma_H0": errors[0],
        "OmegaLambda": omega_lambda,
        "sigma_OmegaLambda": errors[1],
        "chi2": np.sum(result.fun**2),
    }


def mu_lcdm_at_redshift(
    z_value,
    h0,
    omega_lambda,
):
    """
    Flat Lambda-CDM theoretical distance modulus at one redshift.
    """

    grid = np.linspace(
        0.0,
        z_value,
        2001,
    )

    e_squared = (
        (1.0 - omega_lambda)
        * (1.0 + grid)**3
        + omega_lambda
    )

    integral = np.trapz(
        1.0 / np.sqrt(e_squared),
        grid,
    )

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + z_value)
        * integral
    )

    return distance_modulus(d_l)


# ============================================================
# Loga model
# ============================================================

def mu_loga(params):
    """
    params[0] = H0
    """

    h0 = params[0]

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + zHEL)
        * np.log(1.0 + z)
    )

    return distance_modulus(d_l)


def fit_loga(mu_data):

    def residuals(params):

        return solve_triangular(
            L,
            mu_data - mu_loga(params),
            lower=True,
            check_finite=False,
        )

    result = least_squares(
        residuals,
        x0=[73.0],
        bounds=(
            [30.0],
            [120.0],
        ),
        method="trf",
        jac="2-point",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=10000,
    )

    if not result.success:
        raise RuntimeError(
            f"Loga fit failed: {result.message}"
        )

    h0 = result.x[0]

    parameter_covariance = np.linalg.inv(
        result.jac.T @ result.jac
    )

    sigma_h0 = np.sqrt(
        parameter_covariance[0, 0]
    )

    return {
        "H0": h0,
        "sigma_H0": sigma_h0,
        "chi2": np.sum(result.fun**2),
    }


def mu_loga_at_redshift(z_value, h0):

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + z_value)
        * np.log(1.0 + z_value)
    )

    return distance_modulus(d_l)


# ============================================================
# Bin assignments
# ============================================================

bin_id = np.digitize(
    z,
    BIN_EDGES,
    right=False,
) - 1

# Include an object exactly at z = 2.3, if present.
bin_id[z == BIN_EDGES[-1]] = N_BINS - 1

valid = (
    (bin_id >= 0)
    & (bin_id < N_BINS)
)

if not np.all(valid):
    bad_redshifts = z[~valid]

    raise ValueError(
        "Some Pantheon+ redshifts lie outside the adopted bins: "
        f"{bad_redshifts}"
    )


# ============================================================
# Covariance-weighted value in one bin
# ============================================================

def inverse_variance_bin(values, redshifts, covariance, mask):

    values_bin = values[mask]
    redshifts_bin = redshifts[mask]

    variance = np.diag(covariance)[mask]
    weights = 1.0 / variance

    weight_sum = np.sum(weights)

    z_mean = np.sum(weights * redshifts_bin) / weight_sum

    value_mean = np.sum(weights * values_bin) / weight_sum

    sigma = 1.0 / np.sqrt(weight_sum)

    return z_mean, value_mean, sigma


# ============================================================
# Bin the Pantheon+ offsets
# ============================================================

def bin_distance_modulus_offsets(mu_data):

    delta_mu = (
        mu_data
        - mu_eds_data()
    )

    rows = []

    for index in range(N_BINS):

        mask = bin_id == index

        n_bin = int(
            np.sum(mask)
        )

        if n_bin == 0:
            raise ValueError(
                f"Bin {index + 1} is empty."
            )

        z_mean, delta_mu_mean, sigma_mu = inverse_variance_bin(
            delta_mu,
            z,
            cov,
            mask,
        )
        
        rows.append({
            "bin": index,
            "N": n_bin,
            "z_mean": z_mean,
            "delta_mu": delta_mu_mean,
            "sigma_mu": sigma_mu,
        })

    return rows


# ============================================================
# Fit the models
# ============================================================

lcdm_original = fit_lcdm(
    mu_original
)

lcdm_corrected = fit_lcdm(
    mu_corrected
)

loga_corrected = fit_loga(
    mu_corrected
)


# ============================================================
# Obtain binned data
# ============================================================

rows_original = bin_distance_modulus_offsets(
    mu_original
)

rows_corrected = bin_distance_modulus_offsets(
    mu_corrected
)


# ============================================================
# Add theoretical predictions
# ============================================================

for row in rows_original:

    z_mean = row["z_mean"]

    mu_eds_mean = mu_eds_at_redshift(
        z_mean
    )

    mu_lcdm_mean = mu_lcdm_at_redshift(
        z_mean,
        lcdm_original["H0"],
        lcdm_original["OmegaLambda"],
    )

    row["delta_mu_lcdm"] = (
        mu_lcdm_mean
        - mu_eds_mean
    )

    row["pull_lcdm"] = (
        row["delta_mu"]
        - row["delta_mu_lcdm"]
    ) / row["sigma_mu"]


for row in rows_corrected:

    z_mean = row["z_mean"]

    mu_eds_mean = mu_eds_at_redshift(
        z_mean
    )

    mu_lcdm_mean = mu_lcdm_at_redshift(
        z_mean,
        lcdm_corrected["H0"],
        lcdm_corrected["OmegaLambda"],
    )

    mu_loga_mean = mu_loga_at_redshift(
        z_mean,
        loga_corrected["H0"],
    )

    row["delta_mu_lcdm"] = (
        mu_lcdm_mean
        - mu_eds_mean
    )

    row["pull_lcdm"] = (
        row["delta_mu"]
        - row["delta_mu_lcdm"]
    ) / row["sigma_mu"]

    row["delta_mu_loga"] = (
        mu_loga_mean
        - mu_eds_mean
    )

    row["pull_loga"] = (
        row["delta_mu"]
        - row["delta_mu_loga"]
    ) / row["sigma_mu"]


# ============================================================
# Bin-label helper
# ============================================================

def make_bin_label(index):

    lower = BIN_EDGES[index]
    upper = BIN_EDGES[index + 1]

    return f"[{lower:.1f},{upper:.1f})"


# ============================================================
# Construct Table D.1
# ============================================================

title_d1 = (
    "Table D.1. Binned original Pantheon+ distance-modulus "
    "offsets relative to the fiducial Einstein-de Sitter model."
)

model_line_d1 = (
    f"H0_EdS = {H0_EDS:.2f} km/s/Mpc; "
    f"H0_LCDM = {lcdm_original['H0']:.2f} km/s/Mpc; "
    f"Omega_Lambda = {lcdm_original['OmegaLambda']:.3f}."
)

header_d1 = (
    f"{'z_bin':>11} "
    f"{'N_SNe':>7} "
    f"{'<z>':>8} "
    f"{'<Delta_mu>':>12} "
    f"{'sigma_mu':>10} "
    f"{'Delta_mu_LCDM':>15} "
    f"{'pull_LCDM':>11}"
)

separator_d1 = "-" * len(header_d1)

lines_d1 = [
    title_d1,
    model_line_d1,
    "",
    header_d1,
    separator_d1,
]

for row in rows_original:

    lines_d1.append(
        f"{make_bin_label(row['bin']):>11} "
        f"{row['N']:>7d} "
        f"{row['z_mean']:>8.3f} "
        f"{row['delta_mu']:>12.3f} "
        f"{row['sigma_mu']:>10.3f} "
        f"{row['delta_mu_lcdm']:>15.3f} "
        f"{row['pull_lcdm']:>+11.2f}"
    )

lines_d1.append(separator_d1)

table_d1 = "\n".join(
    lines_d1
)


# ============================================================
# Construct Table D.2
# ============================================================

title_d2 = (
    "Table D.2. Binned progenitor-age-corrected Pantheon+ "
    "distance-modulus offsets relative to the fiducial "
    "Einstein-de Sitter model."
)

model_line_d2 = (
    f"H0_EdS = {H0_EDS:.2f} km/s/Mpc; "
    f"H0_LCDM = {lcdm_corrected['H0']:.2f} km/s/Mpc; "
    f"Omega_Lambda = {lcdm_corrected['OmegaLambda']:.3f}; "
    f"H0_Loga = {loga_corrected['H0']:.2f} km/s/Mpc."
)

header_d2 = (
    f"{'z_bin':>11} "
    f"{'N_SNe':>7} "
    f"{'<z>':>8} "
    f"{'<Delta_mu>':>12} "
    f"{'sigma_mu':>10} "
    f"{'Delta_mu_LCDM':>15} "
    f"{'pull_LCDM':>11} "
    f"{'Delta_mu_Loga':>15} "
    f"{'pull_Loga':>10}"
)

separator_d2 = "-" * len(header_d2)

lines_d2 = [
    title_d2,
    model_line_d2,
    "",
    header_d2,
    separator_d2,
]

for row in rows_corrected:

    lines_d2.append(
        f"{make_bin_label(row['bin']):>11} "
        f"{row['N']:>7d} "
        f"{row['z_mean']:>8.3f} "
        f"{row['delta_mu']:>12.3f} "
        f"{row['sigma_mu']:>10.3f} "
        f"{row['delta_mu_lcdm']:>15.3f} "
        f"{row['pull_lcdm']:>+11.2f} "
        f"{row['delta_mu_loga']:>15.3f} "
        f"{row['pull_loga']:>+10.2f}"
    )

lines_d2.append(separator_d2)

table_d2 = "\n".join(
    lines_d2
)


# ============================================================
# Print tables
# ============================================================

print()
print(table_d1)
print()

print(table_d2)
print()


# ============================================================
# Save table text files
# ============================================================

with open(
    "Table_D1.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(table_d1)
    file.write("\n")


with open(
    "Table_D2.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(table_d2)
    file.write("\n")


# ============================================================
# Save compact outputs for Figure 1
# ============================================================

figure_d1 = np.array([
    [
        row["z_mean"],
        row["delta_mu"],
        row["sigma_mu"],
    ]
    for row in rows_original
])

np.savetxt(
    "Figure_1_D1_original.txt",
    figure_d1,
    fmt=[
        "%.10f",
        "%.10f",
        "%.10f",
    ],
    header=(
        "z_mean "
        "Delta_mu "
        "sigma_Delta_mu"
    ),
    comments="",
)


figure_d2 = np.array([
    [
        row["z_mean"],
        row["delta_mu"],
        row["sigma_mu"],
    ]
    for row in rows_corrected
])

np.savetxt(
    "Figure_1_D2_age_corrected.txt",
    figure_d2,
    fmt=[
        "%.10f",
        "%.10f",
        "%.10f",
    ],
    header=(
        "z_mean "
        "Delta_mu "
        "sigma_Delta_mu"
    ),
    comments="",
)


# ============================================================
# Completion message
# ============================================================

print("Saved Table_D1.txt")
print("Saved Table_D2.txt")
print("Saved Figure_1_D1_original.txt")
print("Saved Figure_1_D2_age_corrected.txt")