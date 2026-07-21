"""
Create Table E.1 from the Joint Pantheon+ and DES fit summaries.

Required input files
--------------------
Loga_Joint_original.txt
QuadraticLoga_Joint_original.txt
LCDM_Joint_original.txt

Loga_Joint_age_corrected.txt
QuadraticLoga_Joint_age_corrected.txt
LCDM_Joint_age_corrected.txt

Output
------
Table_E1.txt

------------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

from pathlib import Path
import numpy as np


# ============================================================
# Input files
# ============================================================

FILES = {
    "original": {
        "loga": Path("Loga.Joint_original.txt"),
        "quadratic": Path("Quadratic_Loga.Joint_original.txt"),
        "lcdm": Path("LCDM.Joint_original.txt"),
    },
    "corrected": {
        "loga": Path("Loga.Joint_age_corrected.txt"),
        "quadratic": Path("Quadratic_Loga.Joint_age_corrected.txt"),
        "lcdm": Path("LCDM.Joint_age_corrected.txt"),
    },
}

OUTPUT_FILE = Path("Table_E1.txt")


# ============================================================
# Read one compact fit-summary file
# ============================================================

def read_summary(filename):
    """
    Read a summary file containing one header row and one
    numerical row.

    Example
    -------
    N_Pan N_DES N H0_Pan sigma_H0_Pan H0_DES sigma_H0_DES chi2 AIC BIC
    1701 1820 3521 72.5562 0.1219 68.7945 0.1511 3384.922 3388.922 3401.255
    """

    if not filename.exists():
        raise FileNotFoundError(
            f"Required input file not found: {filename}"
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
            f"{filename} must contain a header row and a data row."
        )

    header = lines[0].split()

    values = np.fromstring(
        lines[1],
        sep=" ",
    )

    if len(header) != len(values):
        raise ValueError(
            f"Column mismatch in {filename}: "
            f"{len(header)} header names but "
            f"{len(values)} numerical values."
        )

    return {
        name: float(value)
        for name, value in zip(
            header,
            values,
        )
    }


# ============================================================
# Flexible key lookup
# ============================================================

def get_value(summary, *possible_keys):
    """
    Return the value corresponding to the first available key.
    This allows minor differences such as OmegaLambda versus
    Omega_Lambda.
    """

    for key in possible_keys:
        if key in summary:
            return summary[key]

    raise KeyError(
        "None of the expected keys were found: "
        + ", ".join(possible_keys)
    )


# ============================================================
# Load all six summaries
# ============================================================

fits = {
    data_state: {
        model: read_summary(filename)
        for model, filename in model_files.items()
    }
    for data_state, model_files in FILES.items()
}


original_loga = fits["original"]["loga"]
original_quadratic = fits["original"]["quadratic"]
original_lcdm = fits["original"]["lcdm"]

corrected_loga = fits["corrected"]["loga"]
corrected_quadratic = fits["corrected"]["quadratic"]
corrected_lcdm = fits["corrected"]["lcdm"]


# ============================================================
# Formatting helpers
# ============================================================

DASH = "—"


def format_value_error(
    value,
    error,
    value_decimals=2,
    error_decimals=2,
):
    return (
        f"{value:.{value_decimals}f} "
        f"± {error:.{error_decimals}f}"
    )


def format_h0(summary, survey):
    """
    Format H0_Pan or H0_DES with its uncertainty.
    """

    if survey == "Pan":
        value = get_value(
            summary,
            "H0_Pan",
            "H0Pan",
        )

        error = get_value(
            summary,
            "sigma_H0_Pan",
            "sigma_H0Pan",
        )

    elif survey == "DES":
        value = get_value(
            summary,
            "H0_DES",
            "H0DES",
        )

        error = get_value(
            summary,
            "sigma_H0_DES",
            "sigma_H0DES",
        )

    else:
        raise ValueError(
            "survey must be 'Pan' or 'DES'."
        )

    return format_value_error(
        value,
        error,
        value_decimals=2,
        error_decimals=2,
    )


def format_delta(summary):
    """
    Quadratic coefficient delta.
    """

    value = get_value(
        summary,
        "delta",
    )

    error = get_value(
        summary,
        "sigma_delta",
    )

    return format_value_error(
        value,
        error,
        value_decimals=3,
        error_decimals=3,
    )


def format_delta_significance(summary):
    """
    Absolute quadratic-coefficient significance |delta|/sigma_delta.
    """

    delta = get_value(
        summary,
        "delta",
    )

    sigma_delta = get_value(
        summary,
        "sigma_delta",
    )

    return f"{abs(delta) / sigma_delta:.2f}"


def format_omega_lambda(summary):
    """
    Format flat-LCDM Omega_Lambda and its uncertainty.

    Accept several possible header conventions used by the
    fitting scripts.
    """

    value = get_value(
        summary,
        "OmegaLambda",
        "Omega_Lambda",
        "Omega_L",
        "OL",
        "omega_lambda",
        "omegaLambda",
        "Omega_DE",
    )

    error = get_value(
        summary,
        "sigma_OmegaLambda",
        "sigma_Omega_Lambda",
        "sigma_Omega_L",
        "sigma_OL",
        "OmegaLambda_err",
        "Omega_Lambda_err",
        "sigma_omega_lambda",
        "sigma_Omega_DE",
    )

    return format_value_error(
        value,
        error,
        value_decimals=3,
        error_decimals=3,
    )


def format_statistic(summary, key):
    return f"{get_value(summary, key):.1f}"


# ============================================================
# Build table values
# ============================================================

columns = [
    original_loga,
    original_quadratic,
    original_lcdm,
    corrected_loga,
    corrected_quadratic,
    corrected_lcdm,
]


rows = [
    (
        "H0_Pan (km/s/Mpc)",
        [
            format_h0(summary, "Pan")
            for summary in columns
        ],
    ),
    (
        "H0_DES (km/s/Mpc)",
        [
            format_h0(summary, "DES")
            for summary in columns
        ],
    ),
    (
        "delta",
        [
            DASH,
            format_delta(original_quadratic),
            DASH,
            DASH,
            format_delta(corrected_quadratic),
            DASH,
        ],
    ),
    (
        "|delta|/sigma_delta",
        [
            DASH,
            format_delta_significance(
                original_quadratic
            ),
            DASH,
            DASH,
            format_delta_significance(
                corrected_quadratic
            ),
            DASH,
        ],
    ),
    (
        "Omega_Lambda",
        [
            DASH,
            DASH,
            format_omega_lambda(
                original_lcdm
            ),
            DASH,
            DASH,
            format_omega_lambda(
                corrected_lcdm
            ),
        ],
    ),
    (
        "chi2_min",
        [
            format_statistic(
                summary,
                "chi2",
            )
            for summary in columns
        ],
    ),
    (
        "AIC",
        [
            format_statistic(
                summary,
                "AIC",
            )
            for summary in columns
        ],
    ),
    (
        "BIC",
        [
            format_statistic(
                summary,
                "BIC",
            )
            for summary in columns
        ],
    ),
]


# ============================================================
# Column widths
# ============================================================

parameter_width = max(
    len("Parameter"),
    max(
        len(label)
        for label, _ in rows
    ),
)

model_names = [
    "Logarithmic",
    "Quadratic",
    "Flat LCDM",
    "Logarithmic",
    "Quadratic",
    "Flat LCDM",
]

column_widths = []

for column_index, model_name in enumerate(
    model_names
):

    width = max(
        len(model_name),
        max(
            len(values[column_index])
            for _, values in rows
        ),
    )

    column_widths.append(
        width + 2
    )


original_group_width = sum(
    column_widths[:3]
)

corrected_group_width = sum(
    column_widths[3:]
)


# ============================================================
# Construct plain-text table
# ============================================================

title_lines = [
    (
        "Table E.1. Joint Pantheon+ and DES fits using the pure "
        "logarithmic relation (delta = 0) and its quadratic "
        "extension, with independent H0 values for the two surveys."
    ),
    (
        "The parameter delta is dimensionless. "
        "The flat LambdaCDM model is shown as a baseline."
    ),
    "",
]


group_header = (
    " " * (parameter_width + 2)
    + "Original data".center(
        original_group_width
    )
    + "Age-corrected data".center(
        corrected_group_width
    )
)


model_header = (
    f"{'Parameter':<{parameter_width}}  "
    + "".join(
        name.center(width)
        for name, width in zip(
            model_names,
            column_widths,
        )
    )
)


separator_length = (
    parameter_width
    + 2
    + sum(column_widths)
)

separator = "-" * separator_length


table_lines = (
    title_lines
    + [
        group_header,
        model_header,
        separator,
    ]
)


for label, values in rows:

    line = (
        f"{label:<{parameter_width}}  "
        + "".join(
            value.center(width)
            for value, width in zip(
                values,
                column_widths,
            )
        )
    )

    table_lines.append(line)


table_lines.append(separator)

table_text = "\n".join(
    table_lines
)


# ============================================================
# Print and save
# ============================================================

print()
print(table_text)
print()


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    file.write(table_text)
    file.write("\n")


print(f"Saved {OUTPUT_FILE}")