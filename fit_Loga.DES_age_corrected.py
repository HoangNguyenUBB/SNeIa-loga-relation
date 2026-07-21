"""
Fit the logarithmic luminosity-distance relation to an individual Type Ia supernova dataset.

Supported datasets
------------------
    DES-SN5YR

Fitted parameter
----------------
    H0

The luminosity distance is

    d_L = (c / H0) (1 + z_hel) ln(1 + z).

When AGE_CORRECTION = 1, the correction

    Delta_mu(z) = 0.183 [1 - exp(-2.2 z)]

is subtracted from the published distance moduli.

-----------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import pandas as pd

from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# User settings
# ============================================================

DATASET = "DES"       # "Pan" or "DES"
AGE_CORRECTION = 1    # 0 : original; 1 : age-corrected

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
# Logarithmic distance modulus
# ============================================================

def mu_loga(params):

    h0 = params[0]

    d_l = (
        (C_LIGHT / h0)
        * (1.0 + z_hel)
        * np.log(1.0 + z)
    )

    return 5.0 * np.log10(d_l) + 25.0


def residuals(params):

    return solve_triangular(
        L,
        mu_data - mu_loga(params),
        lower=True,
        check_finite=False,
    )


# ============================================================
# Fit H0
# ============================================================

result = least_squares(
    residuals,
    x0=[73.0],
    bounds=(
        [40.0],
        [100.0],
    ),
    method="trf",
    jac="2-point",
    xtol=1.0e-12,
    ftol=1.0e-12,
    gtol=1.0e-12,
)


# ============================================================
# Best-fit value and uncertainty
# ============================================================

h0 = result.x[0]

chi2 = np.sum(result.fun**2)

N = len(mu_data)
n_params = 1
dof = N - n_params
reduced_chi2 = chi2 / dof

parameter_covariance = np.linalg.inv(
    result.jac.T @ result.jac
)

sigma_h0 = np.sqrt(
    parameter_covariance[0, 0]
)

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
    f"Loga.{DATASET}_{data_label}.txt"
)


# ============================================================
# Report results
# ============================================================

print()
print(f"Loga | {DATASET} | {display_label}")
print()

print(f"n_params     = {n_params}")
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
    ],
    header=(
        "N "
        "H0 sigma_H0 "
        "chi2 AIC BIC"
    ),
    comments="",
)

print()
print(f"Saved {output_file}")