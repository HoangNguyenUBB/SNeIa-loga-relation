"""
Fit flat Lambda-CDM to an individual Type Ia supernova dataset.

Supported datasets
------------------
    DES-SN5YR

Fitted parameters
-----------------
    Omega_Lambda
    H0

The luminosity distance is

    d_L = (c / H0) (1 + z_hel) integral_0^z dz' / E(z'),

where

    E(z) = sqrt[(1 - Omega_Lambda)(1 + z)^3 + Omega_Lambda].

When AGE_CORRECTION = 1, the correction

    Delta_mu(z) = 0.183 [1 - exp(-2.2 z)]

is subtracted from the published distance moduli.

------------------
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
    
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# User settings
# ============================================================

DATASET = "DES"       # "Pan" or "DES"
AGE_CORRECTION = 0    # 0 : original; 0 : age-corrected

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
    )

    z = df["zHD"].values
    z_hel = df["zHEL"].values
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
    z_hel = df["zHEL"].astype(float).values
    mu_data = df["MU"].astype(float).values

    data = np.load("STAT+SYS.npz")

    N_DES = int(data[data.files[0]][0])
    upper = data[data.files[1]]

    inv_cov = np.zeros((N_DES, N_DES))
    inv_cov[np.triu_indices(N_DES)] = upper

    lower = np.tril_indices(N_DES, -1)
    inv_cov[lower] = inv_cov.T[lower]

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
# Cholesky factor
# ============================================================

L = cho_factor(
    cov,
    lower=True,
    check_finite=False,
)[0]


# ============================================================
# Integration grid
# ============================================================

z_grid = np.linspace(
    0.0,
    np.max(z) * 1.001,
    6000,
)


# ============================================================
# Flat Lambda-CDM distance modulus
# ============================================================

def mu_lcdm(params):

    omega_lambda, h0 = params

    E = np.sqrt(
        (1.0 - omega_lambda)
        * (1.0 + z_grid)**3
        + omega_lambda
    )

    chi_grid = cumtrapz(
        1.0 / E,
        z_grid,
        initial=0.0,
    )

    chi = np.interp(
        z,
        z_grid,
        chi_grid,
    )

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + z_hel)
        * chi
    )

    return 5.0 * np.log10(d_l) + 25.0


def residuals(params):

    return solve_triangular(
        L,
        mu_data - mu_lcdm(params),
        lower=True,
        check_finite=False,
    )


# ============================================================
# Fit Omega_Lambda and H0
# ============================================================

result = least_squares(
    residuals,
    x0=[0.5, 73.0],
    bounds=(
        [0.0, 40.0],
        [1.0, 100.0],
    ),
    method="trf",
    jac="2-point",
    xtol=1.0e-12,
    ftol=1.0e-12,
    gtol=1.0e-12,
)


omega_lambda, h0 = result.x

chi2 = np.sum(result.fun**2)

N = len(mu_data)
n_parameters = 2
dof = N - n_parameters

parameter_covariance = np.linalg.inv(
    result.jac.T @ result.jac
)

sigma_omega_lambda = np.sqrt(
    parameter_covariance[0, 0]
)

sigma_h0 = np.sqrt(
    parameter_covariance[1, 1]
)

reduced_chi2 = chi2 / dof
aic = chi2 + 2.0 * n_parameters
bic = chi2 + n_parameters * np.log(N)


# ============================================================
# Report results
# ============================================================

data_label = (
    "age_corrected"
    if AGE_CORRECTION
    else "original"
)

print()
print(
    f"LCDM | {DATASET} | "
    f"{'Age-corrected' if AGE_CORRECTION else 'Original'}"
)
print()

print(f"n_params     = {n_parameters}")
print(
    f"Omega_Lambda = {omega_lambda:.8f}"
    f" +/- {sigma_omega_lambda:.8f}"
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
# Save summary required by Table 1
# ============================================================

output_file = (
    f"LCDM.{DATASET}_{data_label}.txt"
)

output = np.array([[
    N,
    omega_lambda,
    sigma_omega_lambda,
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
    ],
    header=(
        "N OmegaLambda sigma_OmegaLambda "
        "H0 sigma_H0 "
        "chi2 AIC BIC"
    ),
    comments="",
)

print()
print(f"Saved {output_file}")