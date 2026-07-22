"""
Create Table C1: redshift-cut dependence of the joint fits.

For each upper-redshift cutoff, this script fits:

1. Loga:
       d_L = (c / H0_dataset) (1 + z_hel) ln(1 + z)

2. Flat Lambda-CDM:
       d_L = (c / H0_dataset) (1 + z_hel)
             integral_0^z dz' / E(z')

       E(z)^2 = (1 - Omega_Lambda)(1 + z)^3
                + Omega_Lambda

Pantheon+ and DES have independent H0 parameters in both fits.

The progenitor-age correction is

    Delta_mu(z) = 0.183 [1 - exp(-2.2 z)].

Output
------
Table_C1.txt

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

from scipy.linalg import block_diag, cholesky, solve_triangular
from scipy.optimize import least_squares


# ============================================================
# Settings
# ============================================================

AGE_CORRECTION = True

Z_MAX_VALUES = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    2.3,
]

C_LIGHT = 299792.458
AGE_DELTA = 0.183
AGE_KAPPA = 2.2


# ============================================================
# Read Pantheon+
# ============================================================

df_pan = pd.read_csv(
    "Pantheon+SH0ES.dat",
    sep=r"\s+",
    comment="#",
    encoding="utf-8",
)

zPan_all = df_pan["zHD"].values
zHEL_Pan_all = df_pan["zHEL"].values
muPan_all = df_pan["MU_SH0ES"].values

covPan_all = np.load(
    "Pantheon+SH0ES_STAT+SYS.npy"
)

assert covPan_all.shape == (
    len(zPan_all),
    len(zPan_all),
)


# ============================================================
# Read DES-SN5YR
# ============================================================

with open("DES-Dovekie_HD.csv", "r") as file:
    columns = (
        file.readline()
        .strip()
        .replace("VARNAMES:", "")
        .split()
    )

df_des = pd.read_csv(
    "DES-Dovekie_HD.csv",
    sep=r"\s+",
    skiprows=1,
    names=["ROWTYPE"] + columns,
    engine="python",
)

df_des = df_des[
    df_des["ROWTYPE"] == "SN:"
].copy()

zDES_all = df_des["zHD"].astype(float).values
zHEL_DES_all = df_des["zHEL"].astype(float).values
muDES_all = df_des["MU"].astype(float).values

des_covariance_data = np.load("STAT+SYS.npz")

N_DES_cov = int(
    des_covariance_data[
        des_covariance_data.files[0]
    ][0]
)

upper_triangle = des_covariance_data[
    des_covariance_data.files[1]
]

inv_cov_DES_all = np.zeros(
    (N_DES_cov, N_DES_cov)
)

inv_cov_DES_all[
    np.triu_indices(N_DES_cov)
] = upper_triangle

lower_indices = np.tril_indices(
    N_DES_cov,
    -1,
)

inv_cov_DES_all[lower_indices] = (
    inv_cov_DES_all.T[lower_indices]
)

covDES_all = np.linalg.inv(
    inv_cov_DES_all
)

assert covDES_all.shape == (
    len(zDES_all),
    len(zDES_all),
)


# ============================================================
# Apply progenitor-age correction
# ============================================================

if AGE_CORRECTION:

    muPan_all = muPan_all - AGE_DELTA * (
        1.0 - np.exp(-AGE_KAPPA * zPan_all)
    )

    muDES_all = muDES_all - AGE_DELTA * (
        1.0 - np.exp(-AGE_KAPPA * zDES_all)
    )


# ============================================================
# Fit one redshift-cut sample
# ============================================================

def fit_redshift_cut(z_max):
    """
    Fit Loga and flat Lambda-CDM to the joint dataset restricted
    to zHD <= z_max.
    """

    # --------------------------------------------------------
    # Apply the redshift cut separately to each dataset
    # --------------------------------------------------------

    mask_pan = zPan_all <= z_max
    mask_des = zDES_all <= z_max

    z_pan = zPan_all[mask_pan]
    zhel_pan = zHEL_Pan_all[mask_pan]
    mu_pan = muPan_all[mask_pan]

    z_des = zDES_all[mask_des]
    zhel_des = zHEL_DES_all[mask_des]
    mu_des = muDES_all[mask_des]

    cov_pan = covPan_all[
        np.ix_(mask_pan, mask_pan)
    ]

    cov_des = covDES_all[
        np.ix_(mask_des, mask_des)
    ]

    n_pan = len(z_pan)
    n_des = len(z_des)
    n_total = n_pan + n_des

    if n_pan == 0 or n_des == 0:
        raise ValueError(
            f"z_max={z_max} leaves one dataset empty."
        )

    # --------------------------------------------------------
    # Construct the joint sample
    # --------------------------------------------------------

    z = np.concatenate((
        z_pan,
        z_des,
    ))

    z_hel = np.concatenate((
        zhel_pan,
        zhel_des,
    ))

    mu_data = np.concatenate((
        mu_pan,
        mu_des,
    ))

    covariance = block_diag(
        cov_pan,
        cov_des,
    )

    dataset_id = np.concatenate((
        np.zeros(n_pan, dtype=int),
        np.ones(n_des, dtype=int),
    ))

    # --------------------------------------------------------
    # Cholesky factor
    # --------------------------------------------------------

    L = cholesky(
        covariance,
        lower=True,
        check_finite=False,
    )

    # ========================================================
    # Loga fit
    # ========================================================

    u = np.log(1.0 + z)

    def mu_loga(params):

        h0_pan, h0_des = params

        h0 = np.where(
            dataset_id == 0,
            h0_pan,
            h0_des,
        )

        d_l = (
            (C_LIGHT / h0)
            * (1.0 + z_hel)
            * u
        )

        return 5.0 * np.log10(d_l) + 25.0

    def residuals_loga(params):

        residual = mu_data - mu_loga(params)

        return solve_triangular(
            L,
            residual,
            lower=True,
            check_finite=False,
        )

    fit_loga = least_squares(
        residuals_loga,
        x0=[
            73.0,
            70.0,
        ],
        bounds=(
            [
                30.0,
                30.0,
            ],
            [
                120.0,
                120.0,
            ],
        ),
        method="trf",
        jac="2-point",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=10000,
    )

    if not fit_loga.success:
        raise RuntimeError(
            f"Loga fit failed at z_max={z_max}: "
            f"{fit_loga.message}"
        )

    h0_pan_loga, h0_des_loga = fit_loga.x

    chi2_loga = np.sum(
        fit_loga.fun**2
    )

    n_params_loga = 2
    dof_loga = n_total - n_params_loga
    reduced_chi2_loga = (
        chi2_loga / dof_loga
    )

    fisher_loga = (
        fit_loga.jac.T @ fit_loga.jac
    )

    try:
        covariance_loga = np.linalg.inv(
            fisher_loga
        )
    except np.linalg.LinAlgError:
        covariance_loga = np.linalg.pinv(
            fisher_loga
        )

    errors_loga = np.sqrt(
        np.diag(covariance_loga)
    )

    sigma_h0_pan_loga = errors_loga[0]
    sigma_h0_des_loga = errors_loga[1]

    aic_loga = (
        chi2_loga
        + 2.0 * n_params_loga
    )

    bic_loga = (
        chi2_loga
        + n_params_loga * np.log(n_total)
    )

    # ========================================================
    # Flat Lambda-CDM fit
    # ========================================================

    z_grid, inverse_index = np.unique(
        np.concatenate(([0.0], z)),
        return_inverse=True,
    )

    z_to_grid = inverse_index[1:]

    def lcdm_integral(omega_lambda):

        e2 = (
            (1.0 - omega_lambda)
            * (1.0 + z_grid)**3
            + omega_lambda
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

    def mu_lcdm(params):

        h0_pan, h0_des, omega_lambda = params

        integral = lcdm_integral(
            omega_lambda
        )

        if integral is None:
            return np.full_like(
                mu_data,
                1.0e30,
                dtype=float,
            )

        h0 = np.where(
            dataset_id == 0,
            h0_pan,
            h0_des,
        )

        d_l = (
            (C_LIGHT / h0)
            * (1.0 + z_hel)
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

    def residuals_lcdm(params):

        residual = mu_data - mu_lcdm(params)

        return solve_triangular(
            L,
            residual,
            lower=True,
            check_finite=False,
        )

    fit_lcdm = least_squares(
        residuals_lcdm,
        x0=[
            h0_pan_loga,
            h0_des_loga,
            0.50,
        ],
        bounds=(
            [
                30.0,
                30.0,
                0.0,
            ],
            [
                120.0,
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

    if not fit_lcdm.success:
        raise RuntimeError(
            f"LCDM fit failed at z_max={z_max}: "
            f"{fit_lcdm.message}"
        )

    (
        h0_pan_lcdm,
        h0_des_lcdm,
        omega_lambda,
    ) = fit_lcdm.x

    chi2_lcdm = np.sum(
        fit_lcdm.fun**2
    )

    n_params_lcdm = 3
    dof_lcdm = n_total - n_params_lcdm
    reduced_chi2_lcdm = (
        chi2_lcdm / dof_lcdm
    )

    fisher_lcdm = (
        fit_lcdm.jac.T @ fit_lcdm.jac
    )

    try:
        covariance_lcdm = np.linalg.inv(
            fisher_lcdm
        )
    except np.linalg.LinAlgError:
        covariance_lcdm = np.linalg.pinv(
            fisher_lcdm
        )

    errors_lcdm = np.sqrt(
        np.diag(covariance_lcdm)
    )

    sigma_h0_pan_lcdm = errors_lcdm[0]
    sigma_h0_des_lcdm = errors_lcdm[1]
    sigma_omega_lambda = errors_lcdm[2]

    aic_lcdm = (
        chi2_lcdm
        + 2.0 * n_params_lcdm
    )

    bic_lcdm = (
        chi2_lcdm
        + n_params_lcdm * np.log(n_total)
    )

    # Positive values favor Loga.
    delta_chi2 = (
        chi2_lcdm - chi2_loga
    )

    delta_aic = (
        aic_lcdm - aic_loga
    )

    delta_bic = (
        bic_lcdm - bic_loga
    )

    return {
        "z_max": z_max,
        "N_Pan": n_pan,
        "N_DES": n_des,
        "N": n_total,

        "H0_Pan_Loga": h0_pan_loga,
        "sigma_H0_Pan_Loga": sigma_h0_pan_loga,
        "H0_DES_Loga": h0_des_loga,
        "sigma_H0_DES_Loga": sigma_h0_des_loga,
        "chi2_Loga": chi2_loga,
        "AIC_Loga": aic_loga,
        "BIC_Loga": bic_loga,

        "OmegaLambda": omega_lambda,
        "sigma_OmegaLambda": sigma_omega_lambda,
        "H0_Pan_LCDM": h0_pan_lcdm,
        "sigma_H0_Pan_LCDM": sigma_h0_pan_lcdm,
        "H0_DES_LCDM": h0_des_lcdm,
        "sigma_H0_DES_LCDM": sigma_h0_des_lcdm,
        "chi2_LCDM": chi2_lcdm,
        "AIC_LCDM": aic_lcdm,
        "BIC_LCDM": bic_lcdm,

        "Delta_chi2": delta_chi2,
        "Delta_AIC": delta_aic,
        "Delta_BIC": delta_bic,
    }


# ============================================================
# Run all redshift cuts
# ============================================================

results = []

for z_max in Z_MAX_VALUES:

    print(
        f"Running z_max = {z_max:.1f} ..."
    )

    results.append(
        fit_redshift_cut(z_max)
    )


# ============================================================
# Construct the displayed table
# ============================================================

title = (
    "Table C1. Redshift-cut dependence of the joint "
    "age-corrected fits"
)

note = (
    "Delta values are LCDM minus Loga; "
    "positive values favor Loga."
)

header = (
    f"{'z_max':>6} "
    f"{'N_Pan':>6} "
    f"{'N_DES':>6} "
    f"{'N':>6} "
    f"{'Omega_L':>22} "
    f"{'chi2_Loga':>12} "
    f"{'chi2_LCDM':>12} "
    f"{'Delta_chi2':>12} "
    f"{'Delta_AIC':>11} "
    f"{'Delta_BIC':>11}"
)

separator = "-" * len(header)

lines = [
    title,
    note,
    "",
    header,
    separator,
]

for row in results:

    omega_text = (
        f"{row['OmegaLambda']:.4f}"
        f" +/- "
        f"{row['sigma_OmegaLambda']:.4f}"
    )

    lines.append(
        f"{row['z_max']:>6.1f} "
        f"{row['N_Pan']:>6d} "
        f"{row['N_DES']:>6d} "
        f"{row['N']:>6d} "
        f"{omega_text:>22} "
        f"{row['chi2_Loga']:>12.3f} "
        f"{row['chi2_LCDM']:>12.3f} "
        f"{row['Delta_chi2']:>12.3f} "
        f"{row['Delta_AIC']:>11.3f} "
        f"{row['Delta_BIC']:>11.3f}"
    )

lines.append(separator)

table_text = "\n".join(lines)


# ============================================================
# Print and save
# ============================================================

print()
print(table_text)
print()

with open(
    "Table_C1.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(table_text)
    file.write("\n")


print("Saved Table_C1.txt")