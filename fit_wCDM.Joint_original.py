"""
Fit the flat wCDM to the JOINT Pantheon+ and DES-SN5YR datasets.

Fitted parameters
-----------------
    The two datasets have independent H0 parameters:
        H0_Pan
        H0_DES
    and share common cosmological parameters:
        Omega_DE
        w

Flatness is imposed:

    Omega_M = 1 - Omega_DE.

The expansion rate is

    E(z)^2 =
        (1 - Omega_DE) (1 + z)^3
        + Omega_DE (1 + z)^[3(1 + w)].

The luminosity distance is

    d_L = (c / H0_dataset) (1 + z_hel)
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

# Compatibility across SciPy versions:
# SciPy >= 1.10 uses cumulative_trapezoid; older versions use cumtrapz.
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

from scipy.linalg import block_diag, cholesky, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# User setting
# ============================================================

AGE_CORRECTION = 0   # 0 : original; 1 : age-corrected

C_LIGHT = 299792.458
AGE_DELTA = 0.183
AGE_KAPPA = 2.2


# ============================================================
# Pantheon+
# ============================================================

df = pd.read_csv(
    "Pantheon+SH0ES.dat",
    sep=r"\s+",
    comment="#",
    encoding="utf-8",
)

zPan = df["zHD"].values
zHEL_Pan = df["zHEL"].values
muPan = df["MU_SH0ES"].values

covPan = np.load(
    "Pantheon+SH0ES_STAT+SYS.npy"
)

assert covPan.shape == (
    len(zPan),
    len(zPan),
)


# ============================================================
# DES-SN5YR
# ============================================================

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

zDES = df["zHD"].astype(float).values
zHEL_DES = df["zHEL"].astype(float).values
muDES = df["MU"].astype(float).values

data = np.load("STAT+SYS.npz")

N_DES_cov = int(data[data.files[0]][0])
upper_triangle = data[data.files[1]]

inv_cov_DES = np.zeros(
    (N_DES_cov, N_DES_cov)
)

inv_cov_DES[
    np.triu_indices(N_DES_cov)
] = upper_triangle

lower_indices = np.tril_indices(
    N_DES_cov,
    -1,
)

inv_cov_DES[lower_indices] = (
    inv_cov_DES.T[lower_indices]
)

covDES = np.linalg.inv(inv_cov_DES)

assert covDES.shape == (
    len(zDES),
    len(zDES),
)


# ============================================================
# Apply progenitor-age correction
# ============================================================

if AGE_CORRECTION:

    muPan = muPan - AGE_DELTA * (
        1.0 - np.exp(-AGE_KAPPA * zPan)
    )

    muDES = muDES - AGE_DELTA * (
        1.0 - np.exp(-AGE_KAPPA * zDES)
    )


# ============================================================
# Joint dataset
# ============================================================

z = np.concatenate((
    zPan,
    zDES,
))

zHEL = np.concatenate((
    zHEL_Pan,
    zHEL_DES,
))

mu_data = np.concatenate((
    muPan,
    muDES,
))

cov = block_diag(
    covPan,
    covDES,
)

dataset_id = np.concatenate((
    np.zeros(len(zPan), dtype=int),
    np.ones(len(zDES), dtype=int),
))

N_Pan = len(zPan)
N_DES = len(zDES)
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
    Return the comoving-distance integral evaluated at all
    observed zHD values.
    """

    one_plus_z = 1.0 + z_grid

    e2 = (
        (1.0 - omega_de) * one_plus_z**3
        + omega_de
        * one_plus_z**(3.0 * (1.0 + w))
    )

    if (
        np.any(e2 <= 0.0)
        or not np.all(np.isfinite(e2))
    ):
        return None

    integral_grid = cumtrapz(
        1.0 / np.sqrt(e2),
        z_grid,
        initial=0.0,
    )

    return integral_grid[z_to_grid]


def mu_wcdm(params):
    """
    params[0] = H0_Pan
    params[1] = H0_DES
    params[2] = Omega_DE
    params[3] = w
    """

    h0_pan, h0_des, omega_de, w = params

    h0 = np.where(
        dataset_id == 0,
        h0_pan,
        h0_des,
    )

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

    if (
        np.any(d_l <= 0.0)
        or not np.all(np.isfinite(d_l))
    ):
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
# Fit H0_Pan, H0_DES, Omega_DE, and w
# ============================================================

result = least_squares(
    residuals,
    x0=[
        73.0,
        70.0,
        0.70,
        -1.0,
    ],
    bounds=(
        [
            30.0,
            30.0,
            0.0,
            -3.0,
        ],
        [
            120.0,
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

h0_pan, h0_des, omega_de, w = result.x

chi2 = np.sum(result.fun**2)

n_params = 4
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

sigma_h0_pan = errors[0]
sigma_h0_des = errors[1]
sigma_omega_de = errors[2]
sigma_w = errors[3]

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
    f"wCDM.Joint_{data_label}.txt"
)


# ============================================================
# Report results
# ============================================================

print()
print(f"wCDM | Joint | {display_label}")
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
    f"H0_Pan       = {h0_pan:.8f}"
    f" +/- {sigma_h0_pan:.8f}"
)
print(
    f"H0_DES       = {h0_des:.8f}"
    f" +/- {sigma_h0_des:.8f}"
)

print()
print(f"N_Pan        = {N_Pan}")
print(f"N_DES        = {N_DES}")
print(f"N            = {N}")
print(f"chi2         = {chi2:.6f}")
print(f"AIC          = {aic:.6f}")
print(f"BIC          = {bic:.6f}")


# ============================================================
# Save summary required by downstream scripts
# ============================================================

output = np.array([[
    N_Pan,
    N_DES,
    N,
    omega_de,
    sigma_omega_de,
    w,
    sigma_w,
    h0_pan,
    sigma_h0_pan,
    h0_des,
    sigma_h0_des,
    chi2,
    aic,
    bic,
]])

np.savetxt(
    output_file,
    output,
    fmt=[
        "%d",
        "%d",
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
        "%.10f",
        "%.10f",
    ],
    header=(
        "N_Pan N_DES N "
        "Omega_DE sigma_Omega_DE "
        "w sigma_w "
        "H0_Pan sigma_H0_Pan "
        "H0_DES sigma_H0_DES "
        "chi2 AIC BIC"
    ),
    comments="",
)

print()
print(f"Saved {output_file}")