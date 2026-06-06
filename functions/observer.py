"""Small history container for policy-iteration diagnostics."""

from __future__ import annotations


class Observer:
    """Store the relative value-function errors over PI iterations."""

    def __init__(self) -> None:
        """Create an empty diagnostic history."""
        self.trueErrorList = []

    def add_error(self, true_error: float) -> None:
        """Append one relative value-function error to the history."""
        self.trueErrorList.append(float(true_error))
