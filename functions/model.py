"""Optimal-control model definitions used by the figure scripts.

The module is deliberately small and currently contains only the controlled
Van der Pol oscillator.  New models can be added by following the same method
layout as ``VanDerPol``.
"""

from __future__ import annotations

import abc
import time

import numpy as np
from scipy import linalg as la
from scipy.integrate import solve_bvp


Array = np.ndarray


class Model(metaclass=abc.ABCMeta):
    """Abstract interface for control-affine benchmark models."""

    def __init__(self, state_weight: float, control_weight: float) -> None:
        """Store the quadratic state and control weights."""
        self.stateWeight   = state_weight
        self.controlWeight = control_weight

    @abc.abstractmethod
    def f(self, x: Array) -> Array:
        """Evaluate the uncontrolled vector field."""
        raise NotImplementedError

    @abc.abstractmethod
    def g(self, x: Array) -> Array:
        """Evaluate the control vector field."""
        raise NotImplementedError

    @abc.abstractmethod
    def jacobi_of_f_transposed_dot_p(self, x: Array, p: Array) -> Array:
        """Evaluate Df(x)^T p for the adjoint equation."""
        raise NotImplementedError

    def solve_open_loop_bvp(
        self,
        start_state: Array,
        end_time: float,
        number_of_eval_points: int,
        verbose: bool = False,
    ) -> tuple[Array, Array, float, Array]:
        """Compute a finite-horizon reference value by a PMP boundary problem."""
        n = start_state.shape[0]

        def ode_system(t: Array, y: Array) -> Array:
            """Evaluate state, adjoint, and accumulated-cost dynamics."""
            x       = y[:n, :]
            p       = y[n : 2 * n, :]
            u       = (-1 / (2 * self.controlWeight)) * self.g(x).T @ p
            x_dot   = self.f(x) + self.g(x) @ u
            p_dot   = -self.jacobi_of_f_transposed_dot_p(x, p) - 2 * x * self.stateWeight
            cost    = -self.stateWeight * np.sum(x**2, axis=0) - self.controlWeight * np.sum(u**2, axis=0)

            return np.r_[x_dot, p_dot, np.atleast_2d(cost)]

        def boundary_conditions(ya: Array, yb: Array) -> Array:
            """Enforce initial state and terminal adjoint/cost conditions."""
            return np.r_[ya[0:n] - start_state, yb[n : 2 * n + 1]]

        start_time = time.time()
        time_span  = np.linspace(0, end_time, number_of_eval_points)
        initial    = np.zeros((2 * n + 1, time_span.size))
        solution   = solve_bvp(
            ode_system,
            boundary_conditions,
            time_span,
            initial,
            max_nodes = 10_000_000,
            tol       = 1e-8,
        )

        if verbose:
            message = (
                f"Solved open-loop control with {solution.x.shape[0]} mesh points, "
                f"maximal residual error {np.amax(np.abs(solution.rms_residuals))}. "
                f"It took {time.time() - start_time:.2f} seconds."
            )
            print(message)

        return solution.y[0:n, :], solution.y[n : 2 * n, :], solution.y[-1, 0], solution.x[:]


class VanDerPol(Model):
    """Controlled two-dimensional Van der Pol oscillator."""

    def __init__(self, state_weight: float, control_weight: float) -> None:
        """Initialize the model and the stabilizing Riccati feedback."""
        super().__init__(state_weight, control_weight)

        linearized_matrix = np.c_[np.r_[0, 1], np.r_[1, 1]]
        control_matrix    = np.atleast_2d(np.array([0, 1])).T

        self.matrixKGain  = la.solve_continuous_are(
            linearized_matrix,
            control_matrix,
            state_weight * np.eye(2),
            control_weight,
            e        = None,
            s        = None,
            balanced = True,
        )

    def f(self, x: Array) -> Array:
        """Evaluate the uncontrolled Van der Pol vector field."""
        return np.array([x[1], -x[0] + x[1] * (1 - x[0]**2)])

    def g(self, x: Array) -> Array:
        """Evaluate the constant input vector field."""
        return np.array([[0, 1]]).T

    def jacobi_of_f_transposed_dot_p(self, x: Array, p: Array) -> Array:
        """Evaluate Df(x)^T p for the Van der Pol oscillator."""
        return np.array([-p[1] - 2 * p[1] * x[0] * x[1], p[0] + p[1] * (1 - x[0]**2)])

    def stable_control(self, x: Array) -> Array:
        """Evaluate the stabilizing Riccati feedback on column-wise states."""
        weighted_state = self.matrixKGain @ x
        feedback       = -(1 / self.controlWeight) * np.sum(self.g(x) * weighted_state, axis=0)

        return np.atleast_2d(feedback)

    def control_from_value_gradient(self, x: Array, value_gradient: Array) -> Array:
        """Evaluate the feedback induced by a value-function gradient."""
        return (-0.5 / self.controlWeight) * (self.g(x).T @ value_gradient)

    def closed_loop_rhs(self, x: Array, control: Array) -> Array:
        """Evaluate f(x) + g(x)u for column-wise states and controls."""
        return self.f(x) + self.g(x) @ control

    def running_cost(self, x: Array, control: Array) -> Array:
        """Evaluate the quadratic running cost on column-wise states."""
        state_cost   = self.stateWeight * np.sum(x**2, axis=0)
        control_cost = self.controlWeight * np.sum(control**2, axis=0)

        return state_cost + control_cost

    def linear_value_function(self, x: Array) -> Array:
        """Evaluate the quadratic value induced by the Riccati matrix."""
        return np.sum(x * (self.matrixKGain @ x), axis=0)
