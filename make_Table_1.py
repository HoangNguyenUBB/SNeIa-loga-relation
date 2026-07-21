"""
Create Table 1 from the age-corrected LCDM and Loga fit summaries.

Required input files
--------------------
LCDM_Pan_age_corrected.txt
LCDM_DES_age_corrected.txt
LCDM_Joint_age_corrected.txt

Loga_Pan_age_corrected.txt
Loga_DES_age_corrected.txt
Loga_Joint_age_corrected.txt

Outputs
-------
Table_1.txt

------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

import numpy as np


# ============================================================
# Input filenames
# ============================================================

FILES = {
    ("Pan", "LCDM"): "LCDM.Pan_age_corrected.txt",
    ("Pan", "Loga"): "Loga.Pan_age_corrected.txt",

    ("DES", "LCDM"): "LCDM.DES_age_corrected.txt",
    ("DES", "Loga"): "Loga.DES_age_corrected.txt",

    ("Joint", "LCDM"): "LCDM.Joint_age_corrected.txt",
    ("Joint", "Loga"): "Loga.Joint_age_corrected.txt",
}


# ============================================================
# Read one-row summary files
# ============================================================

def read_summary(filename):
    """
    Read a one-row numerical summary file.

    The function checks the header against the numerical row.
    If an older output file has an incomplete header, the column
    names are reconstructed from the established file format.
    """

    with open(filename, "r", encoding="utf-8") as file:
        header = file.readline().strip().lstrip("#").split()

    values = np.loadtxt(
        filename,
        skiprows=1,
        ndmin=2,
    )

    if values.shape[0] != 1:
        raise ValueError(
            f"{filename} should contain exactly one numerical row."
        )

    values = values[0]
    n_columns = len(values)

    # Use the existing header when it is complete.
    if len(header) == n_columns:
        column_names = header

    # Otherwise reconstruct the expected repository format.
    elif "LCDM_Joint" in filename and n_columns == 12:
        column_names = [
            "N_Pan",
            "N_DES",
            "N",
            "OmegaLambda",
            "sigma_OmegaLambda",
            "H0_Pan",
            "sigma_H0_Pan",
            "H0_DES",
            "sigma_H0_DES",
            "chi2",
            "AIC",
            "BIC",
        ]

    elif "Loga_Joint" in filename and n_columns == 10:
        column_names = [
            "N_Pan",
            "N_DES",
            "N",
            "H0_Pan",
            "sigma_H0_Pan",
            "H0_DES",
            "sigma_H0_DES",
            "chi2",
            "AIC",
            "BIC",
        ]

    elif "LCDM_" in filename and n_columns == 8:
        column_names = [
            "N",
            "OmegaLambda",
            "sigma_OmegaLambda",
            "H0",
            "sigma_H0",
            "chi2",
            "AIC",
            "BIC",
        ]

    elif "Loga_" in filename and n_columns == 6:
        column_names = [
            "N",
            "H0",
            "sigma_H0",
            "chi2",
            "AIC",
            "BIC",
        ]

    else:
        raise ValueError(
            f"Cannot identify the format of {filename}.\n"
            f"Header has {len(header)} columns, but the data row "
            f"has {n_columns} values.\n"
            f"Header: {header}"
        )

    return dict(zip(column_names, values))

# ============================================================
# Formatting helpers
# ============================================================

def value_error(value, error, decimals=3):
    """
    Format a parameter as value +/- uncertainty.
    """

    return (
        f"{value:.{decimals}f} "
        f"+/- {error:.{decimals}f}"
    )

def dash():
    return "--"


# ============================================================
# Load all fit summaries
# ============================================================

results = {}

for key, filename in FILES.items():
    results[key] = read_summary(filename)


# ============================================================
# Construct rows
# ============================================================

rows = []

for dataset in ("Pan", "DES", "Joint"):

    # --------------------------------------------------------
    # Loga
    # --------------------------------------------------------

    loga = results[(dataset, "Loga")]

    if dataset == "Pan":

        h0_pan_text = value_error(
            loga["H0"],
            loga["sigma_H0"],
        )

        h0_des_text = dash()

    elif dataset == "DES":

        h0_pan_text = dash()

        h0_des_text = value_error(
            loga["H0"],
            loga["sigma_H0"],
        )

    else:

        h0_pan_text = value_error(
            loga["H0_Pan"],
            loga["sigma_H0_Pan"],
        )

        h0_des_text = value_error(
            loga["H0_DES"],
            loga["sigma_H0_DES"],
        )

    rows.append({
        "dataset": dataset,
        "model": "Loga",
        "h0_pan": h0_pan_text,
        "h0_des": h0_des_text,
        "omega": dash(),
        "chi2": loga["chi2"],
        "aic": loga["AIC"],
        "bic": loga["BIC"],
    })

    # --------------------------------------------------------
    # LCDM
    # --------------------------------------------------------

    lcdm = results[(dataset, "LCDM")]

    if dataset == "Pan":

        omega_text = value_error(
            lcdm["OmegaLambda"],
            lcdm["sigma_OmegaLambda"],
        )

        h0_pan_text = value_error(
            lcdm["H0"],
            lcdm["sigma_H0"],
        )

        h0_des_text = dash()

    elif dataset == "DES":

        omega_text = value_error(
            lcdm["OmegaLambda"],
            lcdm["sigma_OmegaLambda"],
        )

        h0_pan_text = dash()

        h0_des_text = value_error(
            lcdm["H0"],
            lcdm["sigma_H0"],
        )

    else:

        omega_text = value_error(
            lcdm["OmegaLambda"],
            lcdm["sigma_OmegaLambda"],
        )

        h0_pan_text = value_error(
            lcdm["H0_Pan"],
            lcdm["sigma_H0_Pan"],
        )

        h0_des_text = value_error(
            lcdm["H0_DES"],
            lcdm["sigma_H0_DES"],
        )

    rows.append({
        "dataset": dataset,
        "model": "LCDM",
        "h0_pan": h0_pan_text,
        "h0_des": h0_des_text,
        "omega": omega_text,
        "chi2": lcdm["chi2"],
        "aic": lcdm["AIC"],
        "bic": lcdm["BIC"],
    })

# ============================================================
# Plain-text table
# ============================================================

header = (
    f"{'Dataset':<8}"
    f"{'Model':<8}"
    f"{'H0_Pan':<24}"
    f"{'H0_DES':<24}"
    f"{'Omega_Lambda':<24}"
    f"{'chi2':>12}"
    f"{'AIC':>12}"
    f"{'BIC':>12}"
)

separator = "-" * len(header)

text_lines = [
    "Table 1. Fits to the age-corrected SNe Ia datasets",
    "",
    header,
    separator,
]

for row in rows:

    text_lines.append(
        f"{row['dataset']:<8}"
        f"{row['model']:<8}"
        f"{row['h0_pan']:<24}"
        f"{row['h0_des']:<24}"
        f"{row['omega']:<24}"
        f"{row['chi2']:>12.3f}"
        f"{row['aic']:>12.3f}"
        f"{row['bic']:>12.3f}"
    )

    if row["model"] == "LCDM":
        text_lines.append(separator)

text_table = "\n".join(text_lines)

print()
print(text_table)
print()

with open(
    "Table_1.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(text_table)
    file.write("\n")

print("Saved Table_1.txt")
