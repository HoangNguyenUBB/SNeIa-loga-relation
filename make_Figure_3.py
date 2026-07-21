"""
make_Figure_3.py

Create Figure 3 from the compact bucket files produced by

    make_Tables_G1_and_G2.py

Plotted quantities
------------------
Open squares:
    Original joint Pantheon+ and DES kernel estimates.

Filled circles:
    Age-corrected joint Pantheon+ and DES kernel estimates.

Curves:
    Flat-LCDM theoretical kernels for selected Omega_Lambda.

The first five rows of each input file are the principal redshift
buckets used in Figure 3. Rows 6--10 are the interlaced diagnostic
buckets and are not plotted here.

Outputs
-------
Figure_3.pdf
Figure_3.png

-----------
Author: Hoang Ky Nguyen
Date  : July 2026
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input and output files
# ============================================================

ORIGINAL_BUCKET_FILE = Path(
    "Figure_3_original_buckets.txt"
)

CORRECTED_BUCKET_FILE = Path(
    "Figure_3_age_corrected_buckets.txt"
)

OUTPUT_PDF = "Figure_3.pdf"
OUTPUT_PNG = "Figure_3.png"


# ============================================================
# Figure settings
# ============================================================

Z_MIN = 0.0
Z_MAX = 3.0

K_MIN = 0.6
K_MAX = 1.4

N_CURVE_POINTS = 1000

OMEGA_LAMBDA_VALUES = [
    0.0,
    0.3,
    0.5,
    0.65,
    0.8,
    1.0,
]


# ============================================================
# Read compact bucket file
# ============================================================

def read_bucket_file(filename):
    """
    Read all ten kernel buckets.

    Rows 1--5:
        principal buckets

    Rows 6--10:
        interlaced buckets
    """

    if not filename.exists():
        raise FileNotFoundError(
            f"Required bucket file not found: {filename}"
        )

    data = np.genfromtxt(
        filename,
        names=True,
        dtype=float,
        encoding="utf-8",
    )

    data = np.atleast_1d(data)

    required_columns = {
        "bin",
        "z_low",
        "z_high",
        "N_SNe",
        "z_mean",
        "K_fit",
        "sigma_K",
        "K_LCDM",
    }

    missing = required_columns - set(data.dtype.names)

    if missing:
        raise ValueError(
            f"{filename} is missing these columns: "
            + ", ".join(sorted(missing))
        )

    if len(data) != 10:
        raise ValueError(
            f"{filename} should contain 10 buckets, "
            f"but {len(data)} were found."
        )

    order = np.argsort(data["bin"])

    return data[order]

# ============================================================
# Flat-LCDM theoretical kernel
# ============================================================

def kernel_lcdm(z, omega_lambda):
    """
    Flat-LCDM logarithmic kernel

        K(z) = (1+z) / E(z),

    where

        E(z) = sqrt[
            (1-OmegaLambda)(1+z)^3
            + OmegaLambda
        ].
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    omega_m = (
        1.0
        - omega_lambda
    )

    e_z = np.sqrt(
        omega_m * (1.0 + z)**3
        + omega_lambda
    )

    return (
        (1.0 + z)
        / e_z
    )


# ============================================================
# Error-bar preparation
# ============================================================

def asymmetric_x_errors(bucket_data):
    """
    Return asymmetric horizontal errors from the actual bucket
    boundaries:

        lower error = <z> - z_low
        upper error = z_high - <z>.
    """

    lower = (
        bucket_data["z_mean"]
        - bucket_data["z_low"]
    )

    upper = (
        bucket_data["z_high"]
        - bucket_data["z_mean"]
    )

    if np.any(lower < 0.0) or np.any(upper < 0.0):
        raise ValueError(
            "At least one mean redshift lies outside its bucket."
        )

    return np.vstack((
        lower,
        upper,
    ))


# ============================================================
# Curve labels
# ============================================================

def place_curve_labels(axis, z_curve):
    """
    Add labels near the theoretical curves.

    Positions are chosen to reproduce the layout of the paper
    figure while keeping the labels clear of the data points.
    """

    label_specs = [
    # ΩΛ    x      y_offset   rotation
    (1.00, 0.2,  0.06, 74),
    (0.80, 2.5,  0.035,-17),
    (0.65, 2.35, 0.035,-17),
    (0.50, 2., 0.035,-20),
    (0.30, 1.65, 0.035,-21),
    (0.00, 1.1, 0.035,-23),
    ]

    for omega_lambda, x_position, y_offset, rotation in label_specs:

        y_position = kernel_lcdm(
            x_position,
            omega_lambda,
        )

        label = (
            rf"$\Omega_\Lambda={omega_lambda:g}$"
        )

        # Preserve the EdS designation used in the figure.
        if np.isclose(
            omega_lambda,
            0.0,
        ):
            label = (
                r"$\Omega_\Lambda=0$ (EdS)"
            )

        axis.text(
            x_position,
            y_position + y_offset,
            label,
            fontsize=8.5,
            rotation=rotation,
            rotation_mode="anchor",
            ha="left",
            va="center",
        )


# ============================================================
# Main
# ============================================================

def main():

    original = read_bucket_file(
        ORIGINAL_BUCKET_FILE
    )

    corrected = read_bucket_file(
        CORRECTED_BUCKET_FILE
    )

    z_curve = np.linspace(
        Z_MIN,
        Z_MAX,
        N_CURVE_POINTS,
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 5.0)
    )

    # --------------------------------------------------------
    # Shade the z >= 1 region
    # --------------------------------------------------------

    axis.axvspan(
        1.0,
        Z_MAX,
        alpha=0.09,
        zorder=0,
    )

    # --------------------------------------------------------
    # Flat-LCDM curves
    # --------------------------------------------------------

    curve_styles = {
        0.0: {
            "linestyle": "--",
            "linewidth": 1.2,
            "color": "mediumseagreen",
        },
        0.3: {
            "linestyle": "--",
            "linewidth": 1.2,
            "color": "sienna",
        },
        0.5: {
            "linestyle": "-",
            "linewidth": 1.5,
            "color": "blue",
        },
        0.65: {
            "linestyle": "--",
            "linewidth": 1.5,
            "color": "red",
        },
        0.8: {
            "linestyle": "--",
            "linewidth": 1.1,
            "color": "gray",
        },
        1.0: {
            "linestyle": "--",
            "linewidth": 1.0,
            "color": "black",
        },
    }

    for omega_lambda in OMEGA_LAMBDA_VALUES:

        style = curve_styles[
            omega_lambda
        ]

        axis.plot(
            z_curve,
            kernel_lcdm(
                z_curve,
                omega_lambda,
            ),
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            color=style["color"],
            zorder=1,
        )

    # --------------------------------------------------------
    # Constant-kernel reference
    # --------------------------------------------------------

    axis.axhline(
        1.0,
        color="black",
        linewidth=1.2,
        zorder=2,
    )

    axis.text(
        2.62,
        1.018,
        r"$K=1$",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # --------------------------------------------------------
    # Original data
    # --------------------------------------------------------

    axis.errorbar(
        original["z_mean"],
        original["K_fit"],
        yerr=original["sigma_K"],
        fmt="s",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
        ecolor="black",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        linestyle="none",
        zorder=5,
        label="Original SNe data",
    )
    # --------------------------------------------------------
    # Age-corrected data
    # --------------------------------------------------------

    axis.errorbar(
        corrected["z_mean"],
        corrected["K_fit"],
        yerr=corrected["sigma_K"],
        fmt="o",
        markersize=4.3,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.8,
        ecolor="black",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        linestyle="none",
        zorder=6,
        label="Age-corrected SNe data",
    )

    # --------------------------------------------------------
    # Curve labels
    # --------------------------------------------------------

    place_curve_labels(
        axis,
        z_curve,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    axis.set_xlim(
        Z_MIN,
        Z_MAX,
    )

    axis.set_ylim(
        K_MIN,
        K_MAX,
    )

    axis.set_xlabel(
        r"Redshift $z$",
        fontsize=12,
    )

    axis.set_ylabel(
        r"$K$",
        fontsize=14,
        rotation=0,
        labelpad=14,
    )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="in",
        top=True,
        right=True,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    axis.grid(
        True,
        color="0.82",
        linewidth=0.6,
        alpha=0.55,
    )

    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        handlelength=1.0,
        handletextpad=0.5,
        borderaxespad=0.55,
        labelspacing=0.35,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")
    print("Caption: Figure 3.")


if __name__ == "__main__":
    main()