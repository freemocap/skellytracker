"""Dense One Euro arithmetic shared by keypoints and crop coordinates."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class OneEuroWorkspace:
    filtered_values: NDArray[np.float64]
    derivative_values: NDArray[np.float64]
    value_delta: NDArray[np.float64]
    filter_alpha: NDArray[np.float64]

    @classmethod
    def create(cls, shape: tuple[int, ...]):
        return cls(*(np.empty(shape, dtype=np.float64) for _ in range(4)))


def one_euro_step(values, previous_values, previous_derivatives, *, dt: float,
                  min_cutoff: float, beta: float, d_cutoff: float,
                  workspace: OneEuroWorkspace | None = None):
    """Return filtered values/derivatives; supplied workspace is reused in place.

    Arrays may have any equal shape. Outputs may alias the previous arrays when
    using a retained workspace; consumers retaining frame data must copy it.
    Missing-value and reacquisition policy belongs to the calling data wrapper.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('One Euro elapsed seconds must be finite and positive')
    if values.shape != previous_values.shape or values.shape != previous_derivatives.shape:
        raise ValueError('One Euro arrays must have identical shapes')
    work = workspace if workspace is not None else OneEuroWorkspace.create(values.shape)
    if work.filtered_values.shape != values.shape:
        raise ValueError('One Euro workspace shape differs from its input')
    alpha_d = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * d_cutoff * dt))
    np.subtract(values, previous_values, out=work.value_delta)
    work.value_delta *= alpha_d / dt
    np.multiply(previous_derivatives, 1.0 - alpha_d, out=work.derivative_values)
    work.derivative_values += work.value_delta
    np.abs(work.derivative_values, out=work.filter_alpha)
    work.filter_alpha *= beta
    work.filter_alpha += min_cutoff
    work.filter_alpha *= 2.0 * np.pi * dt
    np.reciprocal(work.filter_alpha, out=work.filter_alpha)
    work.filter_alpha += 1.0
    np.reciprocal(work.filter_alpha, out=work.filter_alpha)
    np.subtract(values, previous_values, out=work.value_delta)
    work.value_delta *= work.filter_alpha
    np.add(previous_values, work.value_delta, out=work.filtered_values)
    return work.filtered_values, work.derivative_values
