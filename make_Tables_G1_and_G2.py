"""
make_Tables_G1_and_G2.py

Create Tables G.1 and G.2 in one run.

Table G.1:
    Original joint Pantheon+ and DES sample.

Table G.2:
    Progenitor-age-corrected joint Pantheon+ and DES sample.

The same read_input_data() routine and input conventions used by
the Joint fitting scripts are retained here.

Outputs
-------
Table_G1.txt
Table_G2.txt
Figure_3_original_buckets.txt
Figure_3_age_corrected_buckets.txt

------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.linalg import block_diag


# ============================================================
# User settings
# ============================================================

C_LIGHT = 299792.458

# Use the same age correction as the fitting scripts.
AGE_DELTA = 0.183
AGE_KAPPA = 2.2

# Joint fit-summary files supplying the survey-specific H0 values.
#
# Original:
#     use the original-data flat-LCDM H0 values.
#
# Age-corrected:
#     use the age-corrected Loga H0 values.

ORIGINAL_H0_FILE = Path(
    "LCDM.Joint_original.txt"
)

CORRECTED_H0_FILE = Path(
    "Loga.Joint_age_corrected.txt"
)

# Best-fit flat-LCDM densities used in Tables G.1 and G.2.

OMEGA_LAMBDA_ORIGINAL = 0.656
OMEGA_LAMBDA_CORRECTED = 0.491


# ============================================================
# Redshift buckets
# ============================================================

MAIN_BINS = [
    (0.000, 0.010),
    (0.010, 0.035),
    (0.035, 0.150),
    (0.150, 0.500),
    (0.500, 2.300),
]

INTERLACED_BINS = [
    (0.008, 0.027),
    (0.027, 0.100),
    (0.100, 0.270),
    (0.270, 0.700),
    (0.700, 2.500),
]

ALL_BINS = (
    MAIN_BINS
    + INTERLACED_BINS
)


# ============================================================
# Age correction
# ============================================================

def age_correction(mu, z, apply_correction):
    """
    Apply

        Delta_mu(z) = AGE_DELTA * [1 - exp(-AGE_KAPPA*z)].

    The correction is subtracted from the observed distance modulus.
    """

    mu_output = np.asarray(
        mu,
        dtype=float,
    ).copy()

    if apply_correction:

        correction = AGE_DELTA * (1.0 - np.exp(-AGE_KAPPA * z))

        mu_output -= correction

    return mu_output


# ============================================================
# Read input data
# ============================================================

def read_input_data(apply_age_correction):
    """
    Read and merge Pantheon+ and DES using exactly the same
    filenames, columns, ordering, and covariance conventions as
    the Joint fitting scripts.

    Returns
    -------
    z
        Hubble-diagram redshift.

    zHEL
        Heliocentric redshift.

    mu_data
        Original or age-corrected distance modulus.

    cov
        Block-diagonal Pantheon+ and DES covariance matrix.

    dataset_id
        0 for Pantheon+ and 1 for DES.

    n_pan, n_des
        Numbers of Pantheon+ and DES supernovae.
    """

    # --------------------------------------------------------
    # Pantheon+
    # --------------------------------------------------------

    pan_file = "Pantheon+SH0ES.dat"

    df_pan = pd.read_csv(
        pan_file,
        sep=r"\s+",
        comment="#",
        encoding="utf-8",
    )

    z_pan = df_pan[
        "zHD"
    ].to_numpy(dtype=float)

    zhel_pan = df_pan[
        "zHEL"
    ].to_numpy(dtype=float)

    mu_pan = df_pan[
        "MU_SH0ES"
    ].to_numpy(dtype=float)

    mu_pan = age_correction(
        mu_pan,
        z_pan,
        apply_age_correction,
    )

    cov_pan = np.load(
        "Pantheon+SH0ES_STAT+SYS.npy"
    )

    if cov_pan.shape != (
        len(z_pan),
        len(z_pan),
    ):
        raise ValueError(
            "Pantheon+ covariance shape does not match "
            "the Pantheon+ data."
        )

    # --------------------------------------------------------
    # DES
    # --------------------------------------------------------

    des_file = "DES-Dovekie_HD.csv"

    with open(
        des_file,
        "r",
        encoding="utf-8",
    ) as file:

        first_line = file.readline().strip()

    columns = (
        first_line
        .replace("VARNAMES:", "")
        .split()
    )

    df_des = pd.read_csv(
        des_file,
        sep=r"\s+",
        skiprows=1,
        names=["ROWTYPE"] + columns,
        engine="python",
    )

    df_des = df_des[
        df_des["ROWTYPE"] == "SN:"
    ].copy()

    z_des = df_des[
        "zHD"
    ].astype(float).to_numpy()

    zhel_des = df_des[
        "zHEL"
    ].astype(float).to_numpy()

    mu_des = df_des[
        "MU"
    ].astype(float).to_numpy()

    mu_des = age_correction(
        mu_des,
        z_des,
        apply_age_correction,
    )

    # DES file stores the upper triangle of the inverse
    # covariance matrix.

    des_covariance_file = np.load(
        "STAT+SYS.npz"
    )

    n_des_from_file = int(
        des_covariance_file[
            des_covariance_file.files[0]
        ][0]
    )

    inverse_covariance_upper = (
        des_covariance_file[
            des_covariance_file.files[1]
        ]
    )

    inverse_covariance_des = np.zeros(
        (
            n_des_from_file,
            n_des_from_file,
        )
    )

    inverse_covariance_des[
        np.triu_indices(n_des_from_file)
    ] = inverse_covariance_upper

    lower_indices = np.tril_indices(
        n_des_from_file,
        -1,
    )

    inverse_covariance_des[
        lower_indices
    ] = inverse_covariance_des.T[
        lower_indices
    ]

    cov_des = np.linalg.inv(
        inverse_covariance_des
    )

    if cov_des.shape != (
        len(z_des),
        len(z_des),
    ):
        raise ValueError(
            "DES covariance shape does not match "
            "the DES data."
        )

    # --------------------------------------------------------
    # Restrict to the range used by the buckets
    # --------------------------------------------------------

    z_max = 2.5

    pan_mask = z_pan <= z_max
    pan_indices = np.where(pan_mask)[0]

    z_pan = z_pan[pan_indices]
    zhel_pan = zhel_pan[pan_indices]
    mu_pan = mu_pan[pan_indices]

    cov_pan = cov_pan[
        np.ix_(
            pan_indices,
            pan_indices,
        )
    ]

    des_mask = z_des <= z_max
    des_indices = np.where(des_mask)[0]

    z_des = z_des[des_indices]
    zhel_des = zhel_des[des_indices]
    mu_des = mu_des[des_indices]

    cov_des = cov_des[
        np.ix_(
            des_indices,
            des_indices,
        )
    ]

    # --------------------------------------------------------
    # Merge without sorting
    # --------------------------------------------------------

    z = np.concatenate((
        z_pan,
        z_des,
    ))

    zhel = np.concatenate((
        zhel_pan,
        zhel_des,
    ))

    mu_data = np.concatenate((
        mu_pan,
        mu_des,
    ))

    cov = block_diag(
        cov_pan,
        cov_des,
    )

    dataset_id = np.concatenate((
        np.zeros(
            len(z_pan),
            dtype=int,
        ),
        np.ones(
            len(z_des),
            dtype=int,
        ),
    ))

    return (
        z,
        zhel,
        mu_data,
        cov,
        dataset_id,
        len(z_pan),
        len(z_des),
    )


# ============================================================
# Read compact Joint fit summary
# ============================================================

def read_summary(filename):
    """
    Read a compact fit-summary file containing one header row
    and one numerical row.
    """

    if not filename.exists():
        raise FileNotFoundError(
            f"Required file not found: {filename}"
        )

    with filename.open(
        "r",
        encoding="utf-8",
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
            and not line.lstrip().startswith("#")
        ]

    if len(lines) < 2:
        raise ValueError(
            f"{filename} must contain a header row "
            "and one numerical row."
        )

    names = lines[0].split()

    values = np.fromstring(
        lines[1],
        sep=" ",
    )

    if len(names) != len(values):
        raise ValueError(
            f"Column mismatch in {filename}: "
            f"{len(names)} names and "
            f"{len(values)} numerical values."
        )

    return dict(
        zip(
            names,
            values,
        )
    )


def normalized_key(name):

    return (
        name
        .replace("_", "")
        .lower()
    )


def get_summary_value(
    summary,
    *possible_names,
):
    """
    Retrieve a value while ignoring underscores and
    capitalization.
    """

    normalized_summary = {
        normalized_key(key): value
        for key, value in summary.items()
    }

    for name in possible_names:

        key = normalized_key(name)

        if key in normalized_summary:
            return float(
                normalized_summary[key]
            )

    raise KeyError(
        "Could not find any of the expected fields: "
        + ", ".join(possible_names)
        + "\nAvailable fields: "
        + ", ".join(summary.keys())
    )


def read_joint_h0(filename):

    summary = read_summary(
        filename
    )

    h0_pan = get_summary_value(
        summary,
        "H0_Pan",
        "H0Pan",
    )

    h0_des = get_summary_value(
        summary,
        "H0_DES",
        "H0DES",
    )

    return h0_pan, h0_des


# ============================================================
# Convert mu to kernel variables
# ============================================================

def make_kernel_data(
    z,
    zhel,
    mu_data,
    cov,
    dataset_id,
    h0_pan,
    h0_des,
):
    """
    Construct

        u = ln(1+z_HD),

        y = H0*dL / [c*(1+z_HEL)].

    The diagonal distance-modulus variances are propagated to
    diagonal y variances.
    """

    h0_row = np.where(
        dataset_id == 0,
        h0_pan,
        h0_des,
    )

    luminosity_distance = 10.0**(
        (mu_data - 25.0) / 5.0
    )

    u = np.log1p(z)

    y = (
        h0_row
        * luminosity_distance
        / (
            C_LIGHT
            * (1.0 + zhel)
        )
    )

    variance_mu = np.diag(
        cov
    )

    sigma_mu = np.sqrt(
        variance_mu
    )

    # dy/dmu = (ln 10 / 5)*y

    sigma_y = (
        np.log(10.0)
        / 5.0
        * y
        * sigma_mu
    )

    variance_y = sigma_y**2

    return {
        "z": z,
        "u": u,
        "y": y,
        "variance_y": variance_y,
    }


# ============================================================
# Weighted kernel fit in one bucket
# ============================================================

def fit_kernel_bucket(
    kernel_data,
    z_low,
    z_high,
    include_upper=False,
):
    """
    Fit the local kernel in one redshift bucket.

    For ordinary buckets:

        y = beta0 + K_fit * u

    For a bucket whose lower edge is exactly zero:

        y = K_fit * u

    so beta0 is fixed to zero.

    The regression uses inverse diagonal y variances.

    The tabulated <z> is the unweighted arithmetic mean.
    """

    z = kernel_data["z"]
    u = kernel_data["u"]
    y = kernel_data["y"]
    variance_y = kernel_data["variance_y"]

    if include_upper:
        mask = (
            (z >= z_low)
            & (z <= z_high)
        )
    else:
        mask = (
            (z >= z_low)
            & (z < z_high)
        )

    n_sne = int(np.count_nonzero(mask))

    if n_sne < 2:
        raise ValueError(
            f"Bucket [{z_low}, {z_high}] contains only "
            f"{n_sne} supernovae."
        )

    z_bin = z[mask]
    u_bin = u[mask]
    y_bin = y[mask]
    variance_bin = variance_y[mask]

    if np.any(variance_bin <= 0.0):
        raise ValueError(
            f"Non-positive variance found in bucket "
            f"[{z_low}, {z_high}]."
        )

    weights = 1.0 / variance_bin

    # Unweighted arithmetic mean redshift
    z_mean = np.mean(z_bin)

    # --------------------------------------------------------
    # First bucket: beta0 fixed to zero
    # --------------------------------------------------------

    if np.isclose(z_low, 0.0):

        denominator = np.sum(
            weights * u_bin**2
        )

        if denominator <= 0.0:
            raise ValueError(
                "Cannot fit zero-intercept kernel: "
                "sum(w*u^2) is non-positive."
            )

        k_fit = np.sum(
            weights * u_bin * y_bin
        ) / denominator

        sigma_k = 1.0 / np.sqrt(
            denominator
        )

        beta0 = 0.0

    # --------------------------------------------------------
    # Other buckets: beta0 free
    # --------------------------------------------------------

    else:

        design_matrix = np.column_stack((
            np.ones(n_sne),
            u_bin,
        ))

        normal_matrix = (
            design_matrix.T
            @ (
                weights[:, None]
                * design_matrix
            )
        )

        right_hand_side = (
            design_matrix.T
            @ (
                weights * y_bin
            )
        )

        parameter_covariance = np.linalg.inv(
            normal_matrix
        )

        parameters = (
            parameter_covariance
            @ right_hand_side
        )

        beta0 = float(
            parameters[0]
        )

        k_fit = float(
            parameters[1]
        )

        sigma_k = float(
            np.sqrt(
                parameter_covariance[1, 1]
            )
        )

    return {
        "N": n_sne,
        "z_mean": z_mean,
        "beta0": beta0,
        "K_fit": k_fit,
        "sigma_K": sigma_k,
    }

# ============================================================
# Flat-LCDM theoretical kernel
# ============================================================

def kernel_lcdm(
    z,
    omega_lambda,
):
    """
    Instantaneous flat-LCDM kernel:

        K_LCDM(z)
          = (1+z) / E(z),

        E(z)
          = sqrt[
              (1-OmegaLambda)(1+z)^3
              + OmegaLambda
            ].
    """

    e_z = np.sqrt(
        (
            1.0
            - omega_lambda
        )
        * (1.0 + z)**3
        + omega_lambda
    )

    return (
        (1.0 + z)
        / e_z
    )


# ============================================================
# Calculate all ten rows
# ============================================================

def calculate_rows(
    kernel_data,
    omega_lambda,
):

    rows = []

    for index, (
        z_low,
        z_high,
    ) in enumerate(
        ALL_BINS,
        start=1,
    ):

        include_upper = (
            index == 5
            or index == 10
        )

        fit = fit_kernel_bucket(
            kernel_data,
            z_low,
            z_high,
            include_upper=include_upper,
        )

        k_theory = kernel_lcdm(
            fit["z_mean"],
            omega_lambda,
        )

        pull_constant = (
            fit["K_fit"] - 1.0
        ) / fit["sigma_K"]

        pull_lcdm = (
            fit["K_fit"]
            - k_theory
        ) / fit["sigma_K"]

        rows.append({
            "bin": index,
            "z_low": z_low,
            "z_high": z_high,
            "N": fit["N"],
            "z_mean": fit["z_mean"],
            "K_fit": fit["K_fit"],
            "sigma_K": fit["sigma_K"],
            "K_LCDM": k_theory,
            "pull_constant": pull_constant,
            "pull_LCDM": pull_lcdm,
        })

    return rows


# ============================================================
# Average and RMS pulls
# ============================================================

def summarize_pulls(rows):

    pull_constant = np.array([
        row["pull_constant"]
        for row in rows
    ])

    pull_lcdm = np.array([
        row["pull_LCDM"]
        for row in rows
    ])

    return {
        "average_constant": np.mean(
            pull_constant
        ),
        "average_lcdm": np.mean(
            pull_lcdm
        ),
        "rms_constant": np.sqrt(
            np.mean(
                pull_constant**2
            )
        ),
        "rms_lcdm": np.sqrt(
            np.mean(
                pull_lcdm**2
            )
        ),
    }


# ============================================================
# Construct plain-text table
# ============================================================

def make_table_text(
    number,
    description,
    omega_lambda,
    rows,
):

    summary = summarize_pulls(
        rows
    )

    title = (
        f"Table G.{number}. Kernel estimates for the ten "
        f"redshift buckets used in Fig. 3 for the {description} "
        "Pantheon+ and DES joint sample."
    )

    subtitle = (
        "The last two columns give the deviations from K = 1 "
        "and the best-fit flat LCDM theoretical prediction "
        f"(OmegaLambda = {omega_lambda:.3f}), respectively."
    )

    header = (
        f"{'Bin':>4} "
        f"{'z_lo':>7} "
        f"{'z_hi':>7} "
        f"{'N_SNe':>7} "
        f"{'<z>':>7} "
        f"{'K_fit':>8} "
        f"{'sigma_K':>8} "
        f"{'K_LCDM':>8} "
        f"{'(K-1)/sK':>10} "
        f"{'(K-KLCDM)/sK':>14}"
    )

    separator = "-" * len(
        header
    )

    lines = [
        title,
        subtitle,
        "",
        header,
        separator,
    ]

    for row in rows:

        if row["bin"] == 6:
            lines.append(
                separator
            )

        lines.append(
            f"{row['bin']:>4d} "
            f"{row['z_low']:>7.3f} "
            f"{row['z_high']:>7.3f} "
            f"{row['N']:>7d} "
            f"{row['z_mean']:>7.3f} "
            f"{row['K_fit']:>8.3f} "
            f"{row['sigma_K']:>8.3f} "
            f"{row['K_LCDM']:>8.3f} "
            f"{row['pull_constant']:>+10.2f} "
            f"{row['pull_LCDM']:>+14.2f}"
        )

    lines.extend([
        separator,
        (
            f"{'Average':>63} "
            f"{summary['average_constant']:>+10.2f} "
            f"{summary['average_lcdm']:>+14.2f}"
        ),
        (
            f"{'RMS':>63} "
            f"{summary['rms_constant']:>10.2f} "
            f"{summary['rms_lcdm']:>14.2f}"
        ),
        separator,
    ])

    return "\n".join(
        lines
    )


# ============================================================
# Save Figure 3 bucket file
# ============================================================

def save_bucket_file(
    filename,
    rows,
):

    output = np.array([
        [
            row["bin"],
            row["z_low"],
            row["z_high"],
            row["N"],
            row["z_mean"],
            row["K_fit"],
            row["sigma_K"],
            row["K_LCDM"],
        ]
        for row in rows
    ])

    np.savetxt(
        filename,
        output,
        fmt=[
            "%d",
            "%.10f",
            "%.10f",
            "%d",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
        ],
        header=(
            "bin z_low z_high N_SNe "
            "z_mean K_fit sigma_K K_LCDM"
        ),
        comments="",
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Read original and age-corrected samples
    # --------------------------------------------------------

    (
        z_original,
        zhel_original,
        mu_original,
        cov_original,
        dataset_id_original,
        n_pan_original,
        n_des_original,
    ) = read_input_data(
        apply_age_correction=False
    )

    (
        z_corrected,
        zhel_corrected,
        mu_corrected,
        cov_corrected,
        dataset_id_corrected,
        n_pan_corrected,
        n_des_corrected,
    ) = read_input_data(
        apply_age_correction=True
    )

    if (
        n_pan_original != n_pan_corrected
        or n_des_original != n_des_corrected
    ):
        raise ValueError(
            "Original and age-corrected sample sizes differ."
        )

    print()
    print(
        f"Merged sample: "
        f"N Pan+ = {n_pan_original}, "
        f"N DES = {n_des_original}, "
        f"N total = "
        f"{n_pan_original + n_des_original}"
    )

    # --------------------------------------------------------
    # Read survey-specific H0 values
    # --------------------------------------------------------

    (
        h0_pan_original,
        h0_des_original,
    ) = read_joint_h0(
        ORIGINAL_H0_FILE
    )

    (
        h0_pan_corrected,
        h0_des_corrected,
    ) = read_joint_h0(
        CORRECTED_H0_FILE
    )

    print()
    print("H0 values used")
    print("--------------")

    print(
        "Original:      "
        f"H0_Pan = {h0_pan_original:.6f}, "
        f"H0_DES = {h0_des_original:.6f}"
    )

    print(
        "Age-corrected: "
        f"H0_Pan = {h0_pan_corrected:.6f}, "
        f"H0_DES = {h0_des_corrected:.6f}"
    )

    # --------------------------------------------------------
    # Kernel variables
    # --------------------------------------------------------

    original_kernel_data = make_kernel_data(
        z=z_original,
        zhel=zhel_original,
        mu_data=mu_original,
        cov=cov_original,
        dataset_id=dataset_id_original,
        h0_pan=h0_pan_original,
        h0_des=h0_des_original,
    )

    corrected_kernel_data = make_kernel_data(
        z=z_corrected,
        zhel=zhel_corrected,
        mu_data=mu_corrected,
        cov=cov_corrected,
        dataset_id=dataset_id_corrected,
        h0_pan=h0_pan_corrected,
        h0_des=h0_des_corrected,
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    rows_g1 = calculate_rows(
        original_kernel_data,
        OMEGA_LAMBDA_ORIGINAL,
    )

    rows_g2 = calculate_rows(
        corrected_kernel_data,
        OMEGA_LAMBDA_CORRECTED,
    )

    table_g1 = make_table_text(
        number=1,
        description="original",
        omega_lambda=OMEGA_LAMBDA_ORIGINAL,
        rows=rows_g1,
    )

    table_g2 = make_table_text(
        number=2,
        description="age-corrected",
        omega_lambda=OMEGA_LAMBDA_CORRECTED,
        rows=rows_g2,
    )

    print()
    print(table_g1)

    print()
    print(table_g2)
    print()

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    Path(
        "Table_G1.txt"
    ).write_text(
        table_g1 + "\n",
        encoding="utf-8",
    )

    Path(
        "Table_G2.txt"
    ).write_text(
        table_g2 + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save compact Figure 3 inputs
    # --------------------------------------------------------

    save_bucket_file(
        "Figure_3_original_buckets.txt",
        rows_g1,
    )

    save_bucket_file(
        "Figure_3_age_corrected_buckets.txt",
        rows_g2,
    )

    print("Saved Table_G1.txt")
    print("Saved Table_G2.txt")
    print(
        "Saved Figure_3_original_buckets.txt"
    )
    print(
        "Saved Figure_3_age_corrected_buckets.txt"
    )


if __name__ == "__main__":
    main()