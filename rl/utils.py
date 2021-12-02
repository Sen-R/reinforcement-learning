"""Utilities module for rl package."""


def soft_update(current, target, alpha):
    """Implements soft updating of `current` towards `target`.

    Implements the following soft-update formula:

    `current := (1 - alpha) * current + alpha * target`

    In the limit `alpha == 0` corresponds to making no change to `current`.
    In the limit `alpha == 1` corresponds to replacing `current` by `target`.
    """
    return (1. - alpha) * current + alpha * target
