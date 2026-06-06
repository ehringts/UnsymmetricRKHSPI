"""Create the relative-error plot for three policy-iteration scenarios.

The script computes reference values for the controlled Van der Pol oscillator,
runs the unsymmetric grid-based policy iteration, and writes the convergence plot
as a PDF.  Reference values are cached under ``data/`` after the first run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

from functions import auxFunctions, plotStyle


OUTPUT_FILE = Path("figures/policy_iteration_errors.pdf")
CACHE_DIR   = Path("data")
ERROR_CACHE = "policy_iteration_errors.npz"


def compute_error_histories(cache_dir: Path = CACHE_DIR, force_recompute: bool = False) -> np.ndarray:
    """Load or compute the three relative-error histories used in the plot."""
    cache_file = cache_dir / ERROR_CACHE

    if cache_file.exists() and not force_recompute:
        data = np.load(cache_file)
        return data["errors"]

    n_iterations = 20
    delta_t      = 5e-7
    errors       = np.zeros((3, n_iterations + 1))

    errors[0, :] = auxFunctions.compute_policy_iteration_case(
        x1_range         = (-1, 1),
        x2_range         = (-0.25, 0.25),
        kernel_parameter = 0.3,
        n_iterations     = n_iterations,
        delta_t          = delta_t,
        shrink_domain    = False,
        cache_file       = cache_dir / "reference_horizontal_domain.npz",
    )
    errors[1, :] = auxFunctions.compute_policy_iteration_case(
        x1_range         = (-0.25, 0.25),
        x2_range         = (-1, 1),
        kernel_parameter = 0.3,
        n_iterations     = n_iterations,
        delta_t          = delta_t,
        shrink_domain    = False,
        cache_file       = cache_dir / "reference_vertical_domain.npz",
    )
    errors[2, :] = auxFunctions.compute_policy_iteration_case(
        x1_range         = (-1, 1),
        x2_range         = (-1, 1),
        kernel_parameter = 0.3,
        n_iterations     = n_iterations,
        delta_t          = delta_t,
        shrink_domain    = True,
        cache_file       = cache_dir / "reference_square_domain.npz",
        shrink_grid_size = 100,
    )

    np.savez(cache_file, errors=errors)

    return errors


def plot_policy_iteration_errors(
    output_file: Path = OUTPUT_FILE,
    cache_dir: Path = CACHE_DIR,
    force_recompute: bool = False,
) -> Path:
    """Create and save the policy-iteration convergence plot."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    errors     = compute_error_histories(cache_dir, force_recompute=force_recompute)
    iterations = np.arange(errors.shape[1])

    with plt.rc_context(plotStyle.rc_parameters()):
        fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)

        ax.plot(iterations, errors[0, :], color="C2", linewidth=1.6, label=r"$\mathbf{(A)}$")
        ax.plot(iterations, errors[1, :], color="C0", linewidth=1.6, label=r"$\mathbf{(B)}$")
        ax.plot(iterations, errors[2, :], color="C1", linewidth=1.6, label=r"$\mathbf{(C)}$")

        ax.set_yscale("log")
        ax.set_xlim(0, iterations[-1])
        ax.set_xticks(iterations)
        ax.set_xlabel(r"Policy iterations")
        ax.set_ylabel(r"Relative error")
        ax.set_ylim(5e-7, 1.5e-1)
        ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-1]))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", alpha=0.6, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(
            loc        = "lower left",
            frameon    = True,
            facecolor  = "white",
            edgecolor  = "black",
            framealpha = 0.3,
        )

        fig.savefig(output_file, bbox_inches="tight", dpi=plotStyle.FIGURE_DPI)
        plt.close(fig)

    return output_file


def main() -> None:
    """Run the script from the command line."""
    path = plot_policy_iteration_errors()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
