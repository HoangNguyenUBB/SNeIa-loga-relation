"""
Fit the flat wCDM luminosity-distance relation to an individual Type Ia supernova dataset.

Supported datasets
------------------
    PantDES-SN5YR

Fitted parameters
-----------------
    H0
    Omega_DE
    w

The dimensionless expansion rate is

    E(z)^2 = (1 - Omega_DE) (1 + z)^3 + Omega_DE (1 + z)^[3(1 + w)].

The luminosity distance is

    d_L = (c / H0) (1 + z_hel)
          integral_0^z dz' / E(z').

When AGE_CORRECTION = 1, the correction

    Delta_mu(z) = 0.183 [1 - exp(-2.2 z)]

is subtracted from the published distance moduli.

-----------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import pandas as pd

from scipy.integrate import cumtrapz
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# User settings
# ============================================================

DATASET = "DES"      # "Pan" or "DES"
AGE_CORRECTION = 0   # 0 : original; 1 : age-corrected

C_LIGHT = 299792.458
AGE_DELTA = 0.183
AGE_KAPPA = 2.2


# ============================================================
# Read data and covariance
# ============================================================

if DATASET == "Pan":

    df = pd.read_csv(
        "Pantheon+SH0ES.dat",
        sep=r"\s+",
        comment="#",
        encoding="utf-8",
    )

    z = df["zHD"].values
    zHEL = df["zHEL"].values
    mu_data = df["MU_SH0ES"].values

    cov = np.load(
        "Pantheon+SH0ES_STAT+SYS.npy"
    )


elif DATASET == "DES":

    with open("DES-Dovekie_HD.csv", "r") as file:
        columns = (
            file.readline()
            .strip()
            .replace("VARNAMES:", "")
            .split()
        )

    df = pd.read_csv(
        "DES-Dovekie_HD.csv",
        sep=r"\s+",
        skiprows=1,
        names=["ROWTYPE"] + columns,
        engine="python",
    )

    df = df[df["ROWTYPE"] == "SN:"].copy()

    z = df["zHD"].astype(float).values
    zHEL = df["zHEL"].astype(float).values
    mu_data = df["MU"].astype(float).values

    data = np.load("STAT+SYS.npz")

    N_DES_cov = int(data[data.files[0]][0])
    upper_triangle = data[data.files[1]]

    inv_cov = np.zeros(
        (N_DES_cov, N_DES_cov)
    )

    inv_cov[
        np.triu_indices(N_DES_cov)
    ] = upper_triangle

    lower_indices = np.tril_indices(
        N_DES_cov,
        -1,
    )

    inv_cov[lower_indices] = (
        inv_cov.T[lower_indices]
    )

    cov = np.linalg.inv(inv_cov)


else:

    raise ValueError(
        "DATASET must be 'Pan' or 'DES'."
    )


assert cov.shape == (
    len(mu_data),
    len(mu_data),
)


# ============================================================
# Apply progenitor-age correction
# ============================================================

if AGE_CORRECTION:

    mu_data = mu_data - AGE_DELTA * (
        1.0 - np.exp(-AGE_KAPPA * z)
    )


# ============================================================
# Dataset size
# ============================================================

N = len(mu_data)


# ============================================================
# Integration grid
# ============================================================

z_grid, inverse_index = np.unique(
    np.concatenate(([0.0], z)),
    return_inverse=True,
)

z_to_grid = inverse_index[1:]


# ============================================================
# Full-covariance whitening
# ============================================================

L = cholesky(
    cov,
    lower=True,
    check_finite=False,
)


# ============================================================
# Flat wCDM luminosity distance
# ============================================================

def wcdm_integral(omega_de, w):
    """
    Return the comoving-distance integral evaluated at each
    observed zHD value.
    """

    one_plus_z = 1.0 + z_grid

    e2 = (
        (1.0 - omega_de) * one_plus_z**3
        + omega_de
        * one_plus_z**(3.0 * (1.0 + w))
    )

    if np.any(e2 <= 0.0) or not np.all(np.isfinite(e2)):
        return None

    integral_grid = cumtrapz(
        1.0 / np.sqrt(e2),
        z_grid,
        initial=0.0,
    )

    return integral_grid[z_to_grid]


def mu_wcdm(params):
    """
    params[0] = H0
    params[1] = Omega_DE
    params[2] = w
    """

    h0, omega_de, w = params

    integral = wcdm_integral(
        omega_de,
        w,
    )

    if integral is None:
        return np.full_like(
            mu_data,
            1.0e30,
            dtype=float,
        )

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + zHEL)
        * integral
    )

    if np.any(d_l <= 0.0) or not np.all(np.isfinite(d_l)):
        return np.full_like(
            mu_data,
            1.0e30,
            dtype=float,
        )

    return 5.0 * np.log10(d_l) + 25.0


def residuals(params):

    residual = mu_data - mu_wcdm(params)

    return solve_triangular(
        L,
        residual,
        lower=True,
        check_finite=False,
    )


# ============================================================
# Fit H0, Omega_DE, and w
# ============================================================

result = least_squares(
    residuals,
    x0=[
        70.0,
        0.70,
        -1.0,
    ],
    bounds=(
        [
            30.0,
            0.0,
            -3.0,
        ],
        [
            120.0,
            2.0,
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
        f"wCDM fit failed: {result.message}"
    )


# ============================================================
# Best-fit values and uncertainties
# ============================================================

h0, omega_de, w = result.x

chi2 = np.sum(result.fun**2)

n_params = 3
dof = N - n_params
reduced_chi2 = chi2 / dof

fisher_matrix = result.jac.T @ result.jac

try:
    parameter_covariance = np.linalg.inv(
        fisher_matrix
    )
except np.linalg.LinAlgError:
    parameter_covariance = np.linalg.pinv(
        fisher_matrix
    )

errors = np.sqrt(
    np.diag(parameter_covariance)
)

sigma_h0 = errors[0]
sigma_omega_de = errors[1]
sigma_w = errors[2]

aic = chi2 + 2.0 * n_params
bic = chi2 + n_params * np.log(N)


# ============================================================
# Labels and output filename
# ============================================================

data_label = (
    "age_corrected"
    if AGE_CORRECTION
    else "original"
)

display_label = (
    "Age-corrected"
    if AGE_CORRECTION
    else "Original"
)

output_file = (
    f"wCDM.{DATASET}_{data_label}.txt"
)


# ============================================================
# Report results
# ============================================================

print()
print(f"wCDM | {DATASET} | {display_label}")
print()

print(f"n_params     = {n_params}")
print(
    f"Omega_DE     = {omega_de:.8f}"
    f" +/- {sigma_omega_de:.8f}"
)
print(
    f"w            = {w:.8f}"
    f" +/- {sigma_w:.8f}"
)
print(
    f"H0           = {h0:.8f}"
    f" +/- {sigma_h0:.8f}"
)

print()
print(f"N            = {N}")
print(f"chi2         = {chi2:.6f}")
print(f"AIC          = {aic:.6f}")
print(f"BIC          = {bic:.6f}")


# ============================================================
# Save summary required by downstream scripts
# ============================================================

output = np.array([[
    N,
    omega_de,
    sigma_omega_de,
    w,
    sigma_w,
    h0,
    sigma_h0,
    chi2,
    aic,
    bic,
]])

np.savetxt(
    output_file,
    output,
    fmt=[
        "%d",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
        "%.10f",
    ],
    header=(
        "N "
        "Omega_DE sigma_Omega_DE "
        "w sigma_w "
        "H0 sigma_H0 "
        "chi2 AIC BIC"
    ),
    comments="",
)

print()
print(f"Saved {output_file}")