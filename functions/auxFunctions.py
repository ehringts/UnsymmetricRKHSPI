"""Auxiliary routines for unsymmetric grid-based policy iteration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import kernel as kernel_module
from .observer import Observer


Array = np.ndarray


def make_cartesian_grid(x1_range: tuple[float, float], x2_range: tuple[float, float], n: int) -> Array:
    """Create a two-dimensional Cartesian grid with states stored by columns."""
    x_values = np.linspace(x1_range[0], x1_range[1], n)
    y_values = np.linspace(x2_range[0], x2_range[1], n)
    X, Y     = np.meshgrid(x_values, y_values)

    return np.array([X.flatten(), Y.flatten()], dtype=np.float64)


def relative_l2_error(reference_values: Array, approximated_values: Array) -> float:
    """Compute the relative discrete L2 error."""
    numerator   = np.sum(np.abs(reference_values - approximated_values) ** 2)
    denominator = np.sum(np.abs(reference_values) ** 2)

    return float(np.sqrt(numerator / denominator))


def compute_reference_values(
    model,
    reference_points: Array,
    cache_file: Path | None = None,
    horizon: float = 50.0,
    number_of_eval_points: int = 1001,
) -> Array:
    """Compute or load reference value-function data for a set of points."""
    if cache_file is not None and cache_file.exists():
        data = np.load(cache_file)
        return data["values"]

    values = np.zeros(reference_points.shape[1])
    for index in range(reference_points.shape[1]):
        _, _, value, _ = model.solve_open_loop_bvp(
            reference_points[:, index],
            horizon,
            number_of_eval_points,
        )
        values[index] = value

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, values=values)

    return values


def unsymmetric_policy_iteration(
    model,
    kernel,
    points: Array,
    n_iterations: int,
    delta_t: float,
    reference_points: Array | None = None,
    reference_values: Array | None = None,
    shrink_domain: bool = False,
    observer: Observer | None = None,
    store_domains: bool = False,
    shrink_grid_size: int = 200,
) -> dict:
    """Run the unsymmetric product-kernel policy iteration.

    The policy evaluation step solves the Euler-discretized linear equation
    by an unsymmetric collocation matrix.  The policy improvement step updates
    the feedback from the gradient of the resulting kernel surrogate.
    """
    if observer is None:
        observer = Observer()

    current_points           = points.copy()
    current_control          = model.stable_control(current_points)
    current_reference_points = None if reference_points is None else reference_points.copy()
    current_reference_values = None if reference_values is None else reference_values.copy()
    domain_data              = []
    shell_points             = None
    level_value              = 0.165
    alpha                    = None
    alpha_centers            = current_points.copy()

    for iteration in range(n_iterations):
        alpha_centers  = current_points.copy()
        shifted_points = current_points + delta_t * model.closed_loop_rhs(current_points, current_control)
        cost_values    = model.running_cost(current_points, current_control)
        collocation    = (
            -kernel_module.product_kernel_matrix(kernel, current_points, shifted_points)
            + kernel_module.product_kernel_matrix(kernel, current_points, current_points)
        )
        alpha          = np.linalg.solve(collocation, delta_t * cost_values)
        gradient       = kernel_module.product_kernel_gradient(kernel, current_points, alpha, current_points)
        new_control    = model.control_from_value_gradient(current_points, gradient)

        if current_reference_points is not None and current_reference_values is not None:
            values = kernel_module.product_kernel_matrix(kernel, current_points, current_reference_points) @ alpha
            error  = relative_l2_error(current_reference_values, values)
            observer.add_error(error)
            print(error)

        if shrink_domain:
            plot_grid = make_cartesian_grid((-1, 1), (-1, 1), shrink_grid_size)
            values    = kernel_module.product_kernel_matrix(kernel, current_points, plot_grid) @ alpha

            if iteration > 0 and shell_points is not None and shell_points.size > 0:
                shell_values = kernel_module.product_kernel_matrix(kernel, current_points, shell_points) @ alpha
                level_value  = float(np.min(shell_values))

            upper_level = np.power(0.99, 1 / (iteration + 1)) * level_value
            lower_level = np.power(0.98, 1 / (iteration + 1)) * level_value

            if store_domains:
                domain_data.append(
                    {
                        "alpha"      : alpha.copy(),
                        "points"     : current_points.copy(),
                        "level"      : upper_level,
                        "grid"       : plot_grid.copy(),
                        "grid_values": values.copy(),
                    }
                )

            shell_points = plot_grid * (values <= upper_level) * (values >= lower_level)
            shell_mask   = ~np.all(shell_points == 0, axis=0)
            shell_points = shell_points[:, shell_mask]

            if current_reference_points is not None and current_reference_values is not None:
                reference_surrogate_values = (
                    kernel_module.product_kernel_matrix(kernel, current_points, current_reference_points) @ alpha
                )
                reference_mask           = reference_surrogate_values <= upper_level
                current_reference_points = current_reference_points[:, reference_mask]
                current_reference_values = current_reference_values[reference_mask]

            point_values    = kernel_module.product_kernel_matrix(kernel, current_points, current_points) @ alpha
            point_mask      = point_values <= upper_level
            current_points  = current_points[:, point_mask]
            current_control = new_control[:, point_mask]
        else:
            current_control = new_control.copy()

    return {
        "alpha"        : alpha,
        "alpha_centers": alpha_centers,
        "control"      : current_control,
        "points"       : current_points,
        "observer"     : observer,
        "domains"      : domain_data,
        "shell_points" : shell_points,
    }


def compute_policy_iteration_case(
    x1_range: tuple[float, float],
    x2_range: tuple[float, float],
    kernel_parameter: float,
    n_iterations: int,
    delta_t: float,
    shrink_domain: bool,
    cache_file: Path | None = None,
    shrink_grid_size: int = 100,
) -> Array:
    """Compute one row of relative errors for the PI convergence plot."""
    from .kernel import QuadMatern
    from .model import VanDerPol

    model            = VanDerPol(1, 1 / 50)
    reference_points = make_cartesian_grid(x1_range, x2_range, 10)
    training_points  = make_cartesian_grid(x1_range, x2_range, 30)
    reference_values = compute_reference_values(model, reference_points, cache_file=cache_file)
    errors           = np.zeros(n_iterations + 1)
    initial_values   = model.linear_value_function(reference_points)
    errors[0]        = relative_l2_error(reference_values, initial_values)
    observer         = Observer()
    pi_kernel        = QuadMatern(kernel_parameter, case=2)

    unsymmetric_policy_iteration(
        model,
        pi_kernel,
        training_points,
        n_iterations,
        delta_t,
        reference_points = reference_points,
        reference_values = reference_values,
        shrink_domain    = shrink_domain,
        observer         = observer,
        shrink_grid_size = shrink_grid_size,
    )
    errors[1:] = np.array(observer.trueErrorList)

    return errors


def select_quasi_centers(number_of_centers: int, input_points: Array) -> Array:
    """Select representative points by a simple farthest-point rule."""
    if input_points.size == 0:
        return input_points

    selected_index  = int(np.argmax(np.sum(input_points * input_points, axis=0)))
    selected_points = np.atleast_2d(input_points[:, selected_index]).T
    min_distances   = None

    for _ in range(max(number_of_centers - 1, 0)):
        value, selected_index, min_distances = _compute_fill_distance(
            input_points,
            selected_points,
            min_distances,
        )
        selected_points = np.c_[selected_points, np.atleast_2d(input_points[:, selected_index]).T]

    return selected_points


def _compute_fill_distance(
    candidate_points: Array,
    selected_points: Array,
    current_min_distances: Array | None,
) -> tuple[float, int, Array]:
    """Compute the farthest candidate from the current selected set."""
    distances = (
        -2 * candidate_points.T.dot(selected_points)
        + np.array([np.sum(candidate_points * candidate_points, axis=0)]).T
        + np.array([np.sum(selected_points * selected_points, axis=0)])
    )

    if current_min_distances is not None:
        distances = np.c_[distances, current_min_distances]

    min_distances = np.min(distances, axis=1)
    index         = int(np.argmax(min_distances))
    value         = float(min_distances[index])

    return value, index, min_distances


def simulate_closed_loop(rhs, initial_state: Array, dt: float, end_time: float) -> Array:
    """Simulate a two-dimensional closed-loop trajectory with explicit Euler."""
    steps              = int(end_time / dt)
    trajectory         = np.zeros((steps, 2))
    state              = initial_state.copy()
    trajectory[0, :]   = state

    for index in range(1, steps):
        state                = state + dt * rhs(state)
        trajectory[index, :] = state

    return trajectory
