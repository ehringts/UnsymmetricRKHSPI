"""Create the nested-domain phase portrait for the shrinking PI variant.

The script runs ten unsymmetric policy-iteration steps with domain shrinking,
plots the resulting nested sublevel-set boundaries, and overlays closed-loop
trajectories starting from the innermost boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import auxFunctions, kernel, model, plotStyle
from functions.kernel import product_kernel_gradient


OUTPUT_FILE = Path("figures/nested_domain_portrait.pdf")


def run_nested_domain_iteration() -> tuple[model.VanDerPol, kernel.QuadMatern, dict]:
    """Run ten shrinking unsymmetric PI steps on the unit square."""
    vdp_model       = model.VanDerPol(1, 1 / 50)
    pi_kernel       = kernel.QuadMatern(0.3, case=2)
    training_points = auxFunctions.make_cartesian_grid((-1, 1), (-1, 1), 30)

    result = auxFunctions.unsymmetric_policy_iteration(
        vdp_model,
        pi_kernel,
        training_points,
        n_iterations  = 10,
        delta_t       = 5e-7,
        shrink_domain = True,
        store_domains    = True,
        shrink_grid_size = 200,
    )

    return vdp_model, pi_kernel, result


def final_closed_loop_rhs(vdp_model: model.VanDerPol, pi_kernel: kernel.QuadMatern, result: dict):
    """Return the final closed-loop vector field induced by the surrogate."""
    alpha   = result["alpha"]
    centers = result["alpha_centers"]

    def rhs(state: np.ndarray) -> np.ndarray:
        """Evaluate the final closed-loop vector field at one state."""
        point    = np.atleast_2d(state).T
        gradient = product_kernel_gradient(pi_kernel, point, alpha, centers)
        control  = vdp_model.control_from_value_gradient(point, gradient)

        return vdp_model.closed_loop_rhs(point, control)[:, 0]

    return rhs


def plot_domain_boundaries(ax, domains: list[dict]) -> None:
    """Plot the stored sublevel-set boundaries."""
    for domain in domains:
        grid        = domain["grid"]
        grid_values = domain["grid_values"]
        side_length = int(np.sqrt(grid.shape[1]))
        X           = grid[0, :].reshape(side_length, side_length)
        Y           = grid[1, :].reshape(side_length, side_length)
        Z           = grid_values.reshape(side_length, side_length)

        ax.contour(X, Y, Z, levels=[domain["level"]], colors="black", linewidths=0.45)


def extract_boundary_from_domain(domain: dict) -> np.ndarray:
    """Extract the longest contour line from a stored domain snapshot."""
    grid        = domain["grid"]
    grid_values = domain["grid_values"]
    side_length = int(np.sqrt(grid.shape[1]))
    X           = grid[0, :].reshape(side_length, side_length)
    Y           = grid[1, :].reshape(side_length, side_length)
    Z           = grid_values.reshape(side_length, side_length)
    fig, ax     = plt.subplots()
    contour     = ax.contour(X, Y, Z, levels=[domain["level"]])
    segments    = contour.allsegs[0]
    plt.close(fig)

    if not segments:
        return np.empty((2, 0))

    longest_segment = max(segments, key=lambda segment: segment.shape[0])

    return longest_segment.T


def select_boundary_points_by_angle(shell_points: np.ndarray, number_of_points: int) -> np.ndarray:
    """Select approximately angle-equidistributed points from a boundary shell."""
    if shell_points.size == 0:
        return shell_points

    angles          = np.arctan2(shell_points[1, :], shell_points[0, :])
    target_angles   = np.linspace(-np.pi, np.pi, number_of_points, endpoint=False)
    selected_points = []

    for angle in target_angles:
        distances = np.abs(np.angle(np.exp(1j * (angles - angle))))
        index     = int(np.argmin(distances))
        selected_points.append(shell_points[:, index])

    return np.array(selected_points).T


def plot_inner_boundary_trajectories(ax, rhs, shell_points: np.ndarray) -> None:
    """Plot trajectories starting from representative inner-boundary points."""
    initial_states = select_boundary_points_by_angle(shell_points, 24)

    for index in range(initial_states.shape[1]):
        trajectory = auxFunctions.simulate_closed_loop(rhs, initial_states[:, index], dt=0.01, end_time=10.0)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="C2", linewidth=0.75, alpha=0.95)


def format_nested_domain_axis(ax) -> None:
    """Apply the shared axis formatting for the nested-domain plot."""
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_nested_domains(output_file: Path = OUTPUT_FILE) -> Path:
    """Create and save the nested-domain phase portrait."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    vdp_model, pi_kernel, result = run_nested_domain_iteration()
    rhs                         = final_closed_loop_rhs(vdp_model, pi_kernel, result)

    with plt.rc_context(plotStyle.rc_parameters()):
        fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)

        boundary_points = extract_boundary_from_domain(result["domains"][-1])

        plot_domain_boundaries(ax, result["domains"])
        plot_inner_boundary_trajectories(ax, rhs, boundary_points)
        format_nested_domain_axis(ax)

        fig.savefig(output_file, bbox_inches="tight", dpi=plotStyle.FIGURE_DPI)
        plt.close(fig)

    return output_file


def main() -> None:
    """Run the script from the command line."""
    path = plot_nested_domains()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
