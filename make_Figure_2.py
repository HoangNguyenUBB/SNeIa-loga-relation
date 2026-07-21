"""
Create Figure 2, upper or lower panels by choosing AGE_CORRECTION = 0 or 1

Outputs
-------
Figure_2_upper_panel.pdf and Figure_2_upper_panel.png
or
Figure_2_lower_panel.pdf and Figure_2_lower_panel.png

---------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================

AGE_CORRECTION = 0    # 0 : Original data (plot upper panel of Figure 2)
                      # 1 : Age-corrected data (plot lower panel of Figure 2)

OMEGA_MIN, OMEGA_MAX = 0.0, 1.4   # Adjust these numbers for Omega_DE bounds
W_MIN, W_MAX = -1.2, 0.0          # Adjust these numbers for w bounds
N_OMEGA, N_W = 181, 141           # Adjust these numbers for resolution


C_LIGHT = 299792.458
AGE_DELTA = 0.183
AGE_KAPPA = 2.2

# The values below are obtained from best fit runs, separately from this batch of scripts
# Values hardcoded here, to be used in plotting
if AGE_CORRECTION:
    # Age-corrected data param values
    ALL_BEST_ODE, ALL_BEST_W = 0.994, -0.359
    PAN_BEST_ODE, PAN_BEST_W = 1.116, -0.296
    DES_BEST_ODE, DES_BEST_W = 0.851, -0.456
    LCDM_ALL_BEST_OL = 0.491
    LCDM_PAN_BEST_OL = 0.462
    LCDM_DES_BEST_OL = 0.513
else:
    # Raw data param values
    ALL_BEST_ODE, ALL_BEST_W = 0.773, -0.750
    PAN_BEST_ODE, PAN_BEST_W = 0.795, -0.695
    DES_BEST_ODE, DES_BEST_W = 0.739, -0.833
    LCDM_ALL_BEST_OL = 0.656
    LCDM_PAN_BEST_OL = 0.638
    LCDM_DES_BEST_OL = 0.670

# --------------------------

def age_correction(mu, z, AGE_CORRECTION):
    mu_corrected = mu.copy()
    if AGE_CORRECTION:
        correction = AGE_DELTA * (1.0 - np.exp(-AGE_KAPPA * z))
        mu_corrected -= correction
    return mu_corrected

# --------------------------
# Load data

# ============================================================
# Pantheon+
# ============================================================

DAT_FILE = "Pantheon+SH0ES.dat"

df = pd.read_csv(DAT_FILE, sep=r"\s+", comment="#", encoding="utf-8")

zPan = df["zHD"].values
zHEL_Pan = df["zHEL"].values
muPan = df["MU_SH0ES"].values

muPan = age_correction(muPan, zPan, AGE_CORRECTION)

COV_FILE = "Pantheon+SH0ES_STAT+SYS.npy"
covPan = np.load(COV_FILE)

assert covPan.shape[0] == len(zPan)

sigmuPan = np.sqrt(np.diag(covPan))

# ============================================================
# DES
# ============================================================

DAT_FILE = "DES-Dovekie_HD.csv"

with open(DAT_FILE, "r") as f:
    first = f.readline().strip()

cols = first.replace("VARNAMES:", "").split()

df = pd.read_csv(
    DAT_FILE,
    sep=r"\s+",
    skiprows=1,
    names=["ROWTYPE"] + cols,
    engine="python"
)

df = df[df["ROWTYPE"] == "SN:"].copy()

zDES = df["zHD"].astype(float).values
zHEL_DES = df["zHEL"].astype(float).values
muDES = df["MU"].astype(float).values

muDES = age_correction(muDES, zDES, AGE_CORRECTION)

COV_FILE = "STAT+SYS.npz"

d = np.load(COV_FILE)
N_from_file = int(d[d.files[0]][0])
upper = d[d.files[1]]

inv_cov_DES = np.zeros((N_from_file, N_from_file))
inv_cov_DES[np.triu_indices(N_from_file)] = upper

i_lower = np.tril_indices(N_from_file, -1)
inv_cov_DES[i_lower] = inv_cov_DES.T[i_lower]

covDES = np.linalg.inv(inv_cov_DES)

assert covDES.shape[0] == len(zDES)

sigmuDES = np.sqrt(np.diag(covDES))


# ============================================================
# Combine and NO SORTING
# ============================================================

# if DATASETS_IN_USE == "All":

from scipy.linalg import block_diag

# Merge arrays: Pan+ first, DES second
z = np.concatenate([zPan, zDES])
zHEL = np.concatenate([zHEL_Pan, zHEL_DES])
mu_data = np.concatenate([muPan, muDES])

# Block diagonal covariance
cov = block_diag(covPan, covDES)

# Dataset label: 0 = Pan+, 1 = DES
dataset_id = np.concatenate([
    np.zeros(len(zPan), dtype=int),
    np.ones(len(zDES), dtype=int)
])


# ============================================================
# Flat wCDM contour for Joint Pantheon+ + DES data
#
# Grid parameters:
#     OmegaDE
#     w
#
# Analytically eliminated nuisance parameters:
#     H0Pan
#     H0DES
#
# Dataset identifiers:
#     idx = 0 : Pantheon+
#     idx = 1 : DES
#
# Flat wCDM:
#     OmegaM = 1 - OmegaDE
#
#     E(z)^2 =
#         OmegaM*(1+z)^3
#         + OmegaDE*(1+z)^[3(1+w)]
#
# Luminosity distance:
#     dL = (c/H0_j)*(1+zHEL)*Integral[dz/E(z)]
#
# Required inputs already loaded:
#     z
#     zHEL
#     mu_data
#     cov
#     idx
#
# The age correction, if used, must already have been applied
# to mu_data before this script is run.
#
# ============================================================

from scipy.linalg import cholesky, solve_triangular

# ------------------------------------------------------------
# 1. Input validation
# ------------------------------------------------------------

idx = dataset_id
z = np.asarray(z, dtype=float)
zHEL = np.asarray(zHEL, dtype=float)
mu_data = np.asarray(mu_data, dtype=float)
cov = np.asarray(cov, dtype=float)
idx = np.asarray(idx, dtype=int)

n_data = len(mu_data)

if z.shape != (n_data,):
    raise ValueError("z and mu_data must have the same length.")

if zHEL.shape != (n_data,):
    raise ValueError("zHEL and mu_data must have the same length.")

if idx.shape != (n_data,):
    raise ValueError("idx and mu_data must have the same length.")

if cov.shape != (n_data, n_data):
    raise ValueError(
        f"cov has shape {cov.shape}; "
        f"expected ({n_data}, {n_data})."
    )

if not np.all(np.isin(idx, [0, 1])):
    raise ValueError(
        "idx must contain only 0 for Pantheon+ and 1 for DES."
    )

if not np.any(idx == 0):
    raise ValueError("No Pantheon+ rows were found.")

if not np.any(idx == 1):
    raise ValueError("No DES rows were found.")

if np.any(z <= 0.0):
    raise ValueError("All cosmological redshifts z must be positive.")

if np.any(zHEL <= -1.0):
    raise ValueError("All zHEL values must satisfy zHEL > -1.")


n_pan = int(np.count_nonzero(idx == 0))
n_des = int(np.count_nonzero(idx == 1))


# ------------------------------------------------------------
# 2. Cholesky factorization
#
# cov = L @ L.T
# ------------------------------------------------------------

L = cholesky(
    cov,
    lower=True,
    check_finite=False
)


# ------------------------------------------------------------
# 3. Design matrix for the two dataset calibrations
#
# Model:
#
#     mu_model = mu_no_H0 + X @ beta
#
# where
#
#     beta_Pan = -5 log10(H0Pan)
#     beta_DES = -5 log10(H0DES)
#
# X has two columns:
#     column 0 = 1 for Pan+, 0 otherwise
#     column 1 = 1 for DES, 0 otherwise
# ------------------------------------------------------------

X = np.column_stack((
    (idx == 0).astype(float),
    (idx == 1).astype(float)
))

# Whiten the design matrix once
WX = solve_triangular(
    L,
    X,
    lower=True,
    check_finite=False
)

# Normal matrix for the two calibration parameters
G = WX.T @ WX
Ginv = np.linalg.inv(G)


# ------------------------------------------------------------
# 4. Fixed numerical integration grid
#
# A fixed grid ensures that the numerical integral is identical
# for Pan+, DES, and merged analyses.
# ------------------------------------------------------------

ZMAX = max(2.6, 1.01*np.max(z))
N_INT = 12000

z_int = np.linspace(
    0.0,
    ZMAX,
    N_INT
)

zp1_int = 1.0 + z_int
dz_int = np.diff(z_int)


# ------------------------------------------------------------
# 5. wCDM distance integral
# ------------------------------------------------------------

def wcdm_integral_at_data(OmegaDE, w):
    """
    Return

        I(z_i) = integral_0^z_i dz'/E(z')

    evaluated at every observed redshift.

    Returns None if E(z)^2 becomes non-positive anywhere
    needed for the integration.
    """

    OmegaM = 1.0 - OmegaDE

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore"
    ):
        E2 = (
            OmegaM * zp1_int**3
            + OmegaDE
            * zp1_int**(3.0 * (1.0 + w))
        )

    if (
        np.any(~np.isfinite(E2))
        or np.any(E2 <= 0.0)
    ):
        return None

    invE = 1.0 / np.sqrt(E2)

    integral_grid = np.empty_like(z_int)
    integral_grid[0] = 0.0

    integral_grid[1:] = np.cumsum(
        0.5
        * (invE[:-1] + invE[1:])
        * dz_int
    )

    return np.interp(
        z,
        z_int,
        integral_grid
    )


# ------------------------------------------------------------
# 6. Distance modulus without H0
#
# Since
#
#     dL = c*(1+zHEL)*I(z)/H0,
#
# we write
#
#     mu = mu_no_H0 - 5 log10(H0).
# ------------------------------------------------------------

def mu_without_H0(OmegaDE, w):

    integral = wcdm_integral_at_data(
        OmegaDE,
        w
    )

    if integral is None:
        return None

    distance_without_H0 = (
        C_LIGHT
        * (1.0 + zHEL)
        * integral
    )

    if (
        np.any(~np.isfinite(distance_without_H0))
        or np.any(distance_without_H0 <= 0.0)
    ):
        return None

    return (
        5.0*np.log10(distance_without_H0)
        + 25.0
    )


# ------------------------------------------------------------
# 7. Analytic calibration profiling
# ------------------------------------------------------------

def profile_H0(OmegaDE, w):
    """
    Returns
    -------
    chi2_profile : float
        Chi-square after analytically optimizing H0Pan and H0DES.

    H0Pan : float
        Profiled Pantheon+ calibration.

    H0DES : float
        Profiled DES calibration.
    """

    mu0 = mu_without_H0(
        OmegaDE,
        w
    )

    if mu0 is None:
        return np.inf, np.nan, np.nan

    y = mu_data - mu0

    # Whiten y
    Wy = solve_triangular(
        L,
        y,
        lower=True,
        check_finite=False
    )

    # GLS solution:
    beta = Ginv @ (WX.T @ Wy)

    # Profiled whitened residual
    r_profile = Wy - WX @ beta

    chi2_profile = float(
        np.dot(r_profile, r_profile)
    )

    # beta_j = -5 log10(H0_j)
    H0Pan = 10.0**(-beta[0]/5.0)
    H0DES = 10.0**(-beta[1]/5.0)

    return chi2_profile, H0Pan, H0DES


# ------------------------------------------------------------
# 8. Parameter grid
#
# These ranges include:
#     LCDM line: w = -1
#     Coasting point: OmegaDE = 1, w = -1/3
#
# Increase N_OMEGA and N_W for final publication output.
# Start with 101 or 121 while testing.
# ------------------------------------------------------------

omega_grid = np.linspace(
    OMEGA_MIN,
    OMEGA_MAX,
    N_OMEGA
)

w_grid = np.linspace(
    W_MIN,
    W_MAX,
    N_W
)

chi2_grid = np.full(
    (N_W, N_OMEGA),
    np.inf
)

H0Pan_grid = np.full_like(
    chi2_grid,
    np.nan
)

H0DES_grid = np.full_like(
    chi2_grid,
    np.nan
)


# ------------------------------------------------------------
# 9. Evaluate grid
# ------------------------------------------------------------

print()
print(
    f"Evaluating {N_OMEGA} x {N_W} "
    f"= {N_OMEGA*N_W} grid points..."
)

for iw, w_value in enumerate(w_grid):

    if iw % 10 == 0:
        print(
            f"  row {iw + 1:3d} of {N_W}", end=" "
        )

    for io, omega_value in enumerate(omega_grid):

        chi2_here, H0Pan_here, H0DES_here = profile_H0(
            omega_value,
            w_value
        )

        chi2_grid[iw, io] = chi2_here
        H0Pan_grid[iw, io] = H0Pan_here
        H0DES_grid[iw, io] = H0DES_here
print()

# ------------------------------------------------------------
# 10. Locate grid minimum
# ------------------------------------------------------------

finite_mask = np.isfinite(chi2_grid)

if not np.any(finite_mask):
    raise RuntimeError(
        "No valid wCDM grid points were found."
    )

flat_best = np.nanargmin(
    np.where(
        finite_mask,
        chi2_grid,
        np.nan
    )
)

iw_best, io_best = np.unravel_index(
    flat_best,
    chi2_grid.shape
)

chi2_min = chi2_grid[iw_best, io_best]
OmegaDE_best = omega_grid[io_best]
w_best = w_grid[iw_best]

H0Pan_best = H0Pan_grid[iw_best, io_best]
H0DES_best = H0DES_grid[iw_best, io_best]

delta_chi2 = chi2_grid - chi2_min


print()
print("Grid best fit")
print("-------------")
print(f"OmegaDE = {OmegaDE_best:.8f}")
print(f"w       = {w_best:.8f}")
print(f"H0Pan   = {H0Pan_best:.8f}")
print(f"H0DES   = {H0DES_best:.8f}")
print(f"chi2    = {chi2_min:.8f}")


# ------------------------------------------------------------
# 11. Evaluate important theoretical points
# ------------------------------------------------------------

# Coasting point
OmegaDE_coasting = 1.0
w_coasting = -1.0/3.0

chi2_coasting, H0Pan_coasting, H0DES_coasting = profile_H0(
    OmegaDE_coasting,
    w_coasting
)

print()
print("Coasting point")
print("-------------------")
print(f"OmegaDE = {OmegaDE_coasting:.8f}")
print(f"w       = {w_coasting:.8f}")
print(f"H0Pan   = {H0Pan_coasting:.8f}")
print(f"H0DES   = {H0DES_coasting:.8f}")
print(f"chi2    = {chi2_coasting:.8f}")
print(
    f"Delta chi2 from grid best = "
    f"{chi2_coasting - chi2_min:.8f}"
)

# ------------------------------------------------------------
# 12. Contour plot
#
# For two fitted shape parameters:
#
#     Delta chi2 = 2.30 : 68.27%
#     Delta chi2 = 6.18 : 95.45%
#     Delta chi2 = 11.83: 99.73%
#     DElta chi2 = 19.33: 99.9937%
# ------------------------------------------------------------

OMEGA_MESH, W_MESH = np.meshgrid(
    omega_grid,
    w_grid
)

levels = [2.30, 6.18, 11.83]
linewidths = [2.0, 1.5, 1.3]
level_labels = ["$1\sigma$", "$2\sigma$", "$3\sigma$"]
level_labels = ["68%", "95%", "99.7%"]
level_colors=["royalblue", "crimson", "darkorange"]

fig, ax = plt.subplots(
    figsize=((1 + np.sqrt(5)) / 2 * 6, 6)
)

# Thicker plot border
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(1.5)

# Tick labels
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Axis labels
ax.set_xlabel(r"$\Omega_{\rm DE}$", fontsize=20, color="black")
ax.set_ylabel(r"$w$", fontsize=20, color="black")

ax.tick_params(
    axis="both",
    which="major",
    labelsize=13,
    colors="black",     # tick marks AND tick labels
    width=1.2,
    length=6
)

contours = plt.contour(
    OMEGA_MESH,
    W_MESH,
    delta_chi2,
    levels=levels,
    colors=level_colors,
    linewidths=linewidths,
)

plot_ODE_best = ALL_BEST_ODE
plot_W_best   = ALL_BEST_W
plot_LCDM_OL_best = LCDM_ALL_BEST_OL

# Best-fit point
plt.scatter(
    plot_ODE_best,  # OmegaDE_best,
    plot_W_best,    # w_best,
    marker="o",
    s=60,
    color="teal",
    zorder=6,
    label=r"$\!\!$w$\,$CDM best fit (Pan+ & DES)"
)

# wCDM point for Pan+ alone
plt.scatter(
    PAN_BEST_ODE,
    PAN_BEST_W,
    marker="x",
    s=60,
    color="black",
    zorder=7,
    label=r"$\!\!$w$\,$CDM best fit (Pan+ only)"
)

#  wCDM point for DES alone
plt.scatter(
    DES_BEST_ODE,
    DES_BEST_W,
    marker="+",
    s=110,
    color="black",
    zorder=7,
    label=r"$\!\!$w$\,$CDM best fit (DES   only)"
)

#  LCDM best point for ALL data
plt.scatter(
    plot_LCDM_OL_best,
    -1,
    marker="s",
    s=55,
    color="red",
    zorder=7,
    label=r"$\!\Lambda\,$CDM best fit (Pan+ & DES)"
)

# Coasting point
plt.scatter(
    1.0,
    -1.0/3.0,
    marker="*",
    s=220,
    color="blue",
    zorder=7,
    label=r"$\!\!d_{\,L}\!=\frac{c}{H_0}\,(1\!+\!z)\,\ln(1\!+\!z)$ relation"+("            " if AGE_CORRECTION else " ")
)

# LCDM locus: w = -1, OmegaLambda free
plt.axhline(
    -1.0,
    linestyle="--",
    linewidth=1.6,
    label=r"$\!\!$flat $\Lambda\,$CDM locus: w$=-1$"
)

# Optional vertical line at OmegaDE = 1
plt.axvline(
    1.0,
    linestyle=":",
    linewidth=1.6
)

plt.xlabel(
    r"$\Omega_{\rm\, DE}$"
)

plt.ylabel(
    r"$w$"
)

plt.xlim(
    OMEGA_MIN,
    OMEGA_MAX
)

plt.ylim(
    W_MIN,
    W_MAX
)

plt.legend(
    frameon=False,
    loc="best"
)

plt.grid(
    alpha=0.22
)

plt.tight_layout()

output_file_pdf = "Figure_2" + ("_lower_panel" if AGE_CORRECTION else "_upper_panel") +".pdf"
output_file_png = "Figure_2" + ("_lower_panel" if AGE_CORRECTION else "_upper_panel") +".png"

plt.savefig(
    output_file_pdf,
    bbox_inches="tight",
)

plt.savefig(
    output_file_png,
    dpi=300,
    bbox_inches="tight"
)

prefix  = "Age-corrected" if AGE_CORRECTION else "Original"

plot_title = f"{prefix} Pantheon+ & DES:"

leg = plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    # frameon=False,
    framealpha=1.0,
    edgecolor="none",
    borderaxespad=0,
    title=plot_title,
    title_fontsize=13,
    fontsize=12
)

for text in leg.get_texts():
    text.set_color("black")
    
leg.get_title().set_fontweight('bold')
plt.show()

print("Saved", output_file_pdf)
print("Saved", output_file_png)