"""Independent analytic reference for the Phase-9 IQ likelihood.

This module deliberately does not import either Phase-9 physics backend.  It
models the readout contract only:

* one latent ancilla label is drawn for an entire readout window;
* conditional on that label, the I/Q samples are independent isotropic
  Gaussians with standard deviation ``sigma`` per component;
* the observation density is the corresponding finite Gaussian mixture.

The implementation uses only the Python standard library.  It is intended as
an independent semantic anchor for backend qualification, not as another
oscillator or qutrit solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import erf, exp, fabs, isfinite, log, pi, sqrt
from typing import Sequence


REFERENCE_ID = "PHASE9-IQ-GAUSSIAN-MIXTURE-REFERENCE-V1"
RAW_BASE_MEASURE = "two_dimensional_lebesgue_per_complex_iq_sample"
SIGMA_CONVENTION = "per_real_axis_standard_deviation"
INTEGRATION_CONVENTION = "arithmetic_mean_over_window"
_PROBABILITY_TOLERANCE = 1.0e-12


def _strict_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(
    values: Sequence[object],
    name: str,
    *,
    minimum_length: int = 1,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    try:
        result = tuple(
            _finite_float(value, f"{name}[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if len(result) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} values")
    return result


def _centers(
    values: Sequence[Sequence[object]],
) -> tuple[tuple[float, float], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("centers must be a sequence")
    result: list[tuple[float, float]] = []
    try:
        for index, row in enumerate(values):
            vector = _finite_vector(
                row,
                f"centers[{index}]",
                minimum_length=2,
            )
            if len(vector) != 2:
                raise ValueError(
                    f"centers[{index}] must contain exactly I and Q"
                )
            result.append((vector[0], vector[1]))
    except TypeError as exc:
        raise TypeError("centers must be a sequence") from exc
    if len(result) < 2:
        raise ValueError("centers must contain at least two labels")
    return tuple(result)


def _probabilities(
    values: Sequence[object],
    *,
    expected_length: int,
) -> tuple[float, ...]:
    result = _finite_vector(values, "priors", minimum_length=2)
    if len(result) != expected_length:
        raise ValueError("priors and centers must have equal lengths")
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("priors must lie in [0,1]")
    if abs(sum(result) - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError("priors must sum to one")
    return result


def _samples(
    iq_i: Sequence[object],
    iq_q: Sequence[object],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    i_values = _finite_vector(iq_i, "iq_i")
    q_values = _finite_vector(iq_q, "iq_q")
    if len(i_values) != len(q_values):
        raise ValueError("iq_i and iq_q must have equal lengths")
    return i_values, q_values


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    result = maximum + log(sum(exp(value - maximum) for value in values))
    if not isfinite(result):
        raise RuntimeError("logsumexp produced a non-finite result")
    return result


def within_window_residual(
    iq_i: Sequence[object],
    iq_q: Sequence[object],
) -> tuple[tuple[float, float], float]:
    """Return the integrated IQ mean and within-window squared residual.

    The residual is ``sum_j ||x_j - x_bar||^2`` in raw IQ units.
    """

    i_values, q_values = _samples(iq_i, iq_q)
    count = len(i_values)
    mean_i = sum(i_values) / count
    mean_q = sum(q_values) / count
    residual = sum(
        (sample_i - mean_i) ** 2 + (sample_q - mean_q) ** 2
        for sample_i, sample_q in zip(i_values, q_values)
    )
    if residual < 0.0 or not isfinite(residual):
        raise RuntimeError("within-window residual is invalid")
    return (mean_i, mean_q), residual


def component_log_likelihoods(
    iq_i: Sequence[object],
    iq_q: Sequence[object],
    *,
    centers: Sequence[Sequence[object]],
    sigma: object,
) -> tuple[float, ...]:
    """Evaluate the complete raw-window log density for every label."""

    i_values, q_values = _samples(iq_i, iq_q)
    means = _centers(centers)
    scale = _positive_float(sigma, "sigma")
    count = len(i_values)
    normalization = -count * log(2.0 * pi * scale * scale)
    denominator = 2.0 * scale * scale
    result = tuple(
        normalization
        - sum(
            (sample_i - center_i) ** 2 + (sample_q - center_q) ** 2
            for sample_i, sample_q in zip(i_values, q_values)
        )
        / denominator
        for center_i, center_q in means
    )
    if not all(isfinite(value) for value in result):
        raise RuntimeError("component likelihood is non-finite")
    return result


def residual_decomposed_log_likelihoods(
    iq_i: Sequence[object],
    iq_q: Sequence[object],
    *,
    centers: Sequence[Sequence[object]],
    sigma: object,
) -> tuple[float, ...]:
    """Evaluate the same likelihood through the sufficient-statistic split.

    For every center ``mu_k``,
    ``sum ||x_j-mu_k||^2 = residual + N ||x_bar-mu_k||^2``.
    Keeping this path separate makes sample-count and ``sigma/sqrt(N)``
    mistakes directly testable.
    """

    i_values, q_values = _samples(iq_i, iq_q)
    means = _centers(centers)
    scale = _positive_float(sigma, "sigma")
    (mean_i, mean_q), residual = within_window_residual(i_values, q_values)
    count = len(i_values)
    normalization = -count * log(2.0 * pi * scale * scale)
    denominator = 2.0 * scale * scale
    result = tuple(
        normalization
        - (
            residual
            + count
            * ((mean_i - center_i) ** 2 + (mean_q - center_q) ** 2)
        )
        / denominator
        for center_i, center_q in means
    )
    if not all(isfinite(value) for value in result):
        raise RuntimeError("decomposed likelihood is non-finite")
    return result


def evidence_and_posterior(
    component_logs: Sequence[object],
    *,
    priors: Sequence[object],
) -> tuple[float, tuple[float, ...]]:
    """Return mixture log evidence and normalized posterior probabilities."""

    logs = _finite_vector(component_logs, "component_logs", minimum_length=2)
    probabilities = _probabilities(priors, expected_length=len(logs))
    weighted_logs = tuple(
        log_value + log(probability)
        if probability > 0.0
        else float("-inf")
        for log_value, probability in zip(logs, probabilities)
    )
    finite_weighted = tuple(value for value in weighted_logs if isfinite(value))
    if not finite_weighted:
        raise RuntimeError("at least one positive-prior component is required")
    evidence = _logsumexp(finite_weighted)
    posterior = tuple(
        exp(value - evidence) if isfinite(value) else 0.0
        for value in weighted_logs
    )
    total = sum(posterior)
    if (
        not all(isfinite(value) and 0.0 <= value <= 1.0 for value in posterior)
        or abs(total - 1.0) > 5.0e-13
    ):
        raise RuntimeError("posterior normalization failed")
    return evidence, posterior


def pairwise_log_likelihood_ratios(
    component_logs: Sequence[object],
) -> tuple[tuple[float, ...], ...]:
    """Return ``log L_i - log L_j`` for every ordered label pair."""

    logs = _finite_vector(component_logs, "component_logs", minimum_length=2)
    result = tuple(
        tuple(left - right for right in logs)
        for left in logs
    )
    if not all(isfinite(value) for row in result for value in row):
        raise RuntimeError("LLR matrix is non-finite")
    return result


def integrated_predictive_moments(
    *,
    priors: Sequence[object],
    centers: Sequence[Sequence[object]],
    sigma: object,
    sample_count: object,
) -> tuple[
    tuple[float, float],
    tuple[tuple[float, float], tuple[float, float]],
]:
    """Return exact mean and covariance of the integrated window IQ.

    The between-label covariance is not divided by the sample count because
    the latent label is shared by every sample in a window.  Only conditional
    Gaussian noise contributes ``sigma^2 / N``.
    """

    means = _centers(centers)
    probabilities = _probabilities(priors, expected_length=len(means))
    scale = _positive_float(sigma, "sigma")
    count = _strict_int(sample_count, "sample_count", minimum=1)
    mean_i = sum(p * center[0] for p, center in zip(probabilities, means))
    mean_q = sum(p * center[1] for p, center in zip(probabilities, means))
    conditional = scale * scale / count
    covariance_ii = conditional + sum(
        p * (center[0] - mean_i) ** 2
        for p, center in zip(probabilities, means)
    )
    covariance_qq = conditional + sum(
        p * (center[1] - mean_q) ** 2
        for p, center in zip(probabilities, means)
    )
    covariance_iq = sum(
        p * (center[0] - mean_i) * (center[1] - mean_q)
        for p, center in zip(probabilities, means)
    )
    return (
        (mean_i, mean_q),
        (
            (covariance_ii, covariance_iq),
            (covariance_iq, covariance_qq),
        ),
    )


def integrated_marginal_cdf(
    value: object,
    *,
    axis: object,
    priors: Sequence[object],
    centers: Sequence[Sequence[object]],
    sigma: object,
    sample_count: object,
) -> float:
    """Return the exact I or Q marginal CDF of the integrated IQ."""

    threshold = _finite_float(value, "value")
    coordinate = _strict_int(axis, "axis", minimum=0)
    if coordinate not in (0, 1):
        raise ValueError("axis must be 0 (I) or 1 (Q)")
    means = _centers(centers)
    probabilities = _probabilities(priors, expected_length=len(means))
    scale = _positive_float(sigma, "sigma")
    count = _strict_int(sample_count, "sample_count", minimum=1)
    integrated_sigma = scale / sqrt(count)
    result = sum(
        probability
        * 0.5
        * (
            1.0
            + erf(
                (threshold - center[coordinate])
                / (integrated_sigma * sqrt(2.0))
            )
        )
        for probability, center in zip(probabilities, means)
    )
    if not isfinite(result) or result < -1.0e-15 or result > 1.0 + 1.0e-15:
        raise RuntimeError("marginal CDF is invalid")
    return min(1.0, max(0.0, result))


def integrated_mean_gain_jacobian(
    *,
    priors: Sequence[object],
    centers: Sequence[Sequence[object]],
) -> tuple[float, float]:
    """Derivative of predictive mean under ``centers -> gain * centers``."""

    means = _centers(centers)
    probabilities = _probabilities(priors, expected_length=len(means))
    return (
        sum(p * center[0] for p, center in zip(probabilities, means)),
        sum(p * center[1] for p, center in zip(probabilities, means)),
    )


def affine_log_density_correction(
    *,
    sample_count: object,
    gain_matrix: Sequence[Sequence[object]],
) -> float:
    """Return the density correction for ``y = G x + b``.

    A complete window contains ``N`` independent two-dimensional samples
    conditional on its shared label.  Relative to the raw-IQ Lebesgue base
    measure, ``log p_y(y) = log p_x(x) - N log |det G|``.  Translation has
    unit Jacobian and therefore does not enter the correction.
    """

    count = _strict_int(sample_count, "sample_count", minimum=1)
    if isinstance(gain_matrix, (str, bytes)):
        raise TypeError("gain_matrix must be a 2x2 sequence")
    try:
        rows = tuple(
            _finite_vector(row, f"gain_matrix[{index}]", minimum_length=2)
            for index, row in enumerate(gain_matrix)
        )
    except TypeError as exc:
        raise TypeError("gain_matrix must be a 2x2 sequence") from exc
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("gain_matrix must be exactly 2x2")
    determinant = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    if not isfinite(determinant) or fabs(determinant) <= 1.0e-15:
        raise ValueError("gain_matrix must be nonsingular")
    correction = -count * log(fabs(determinant))
    if not isfinite(correction):
        raise RuntimeError("affine density correction is non-finite")
    return correction


def per_complex_sample_log_score(
    log_evidence: object,
    *,
    sample_count: object,
) -> float:
    """Normalize a complete-window log score to nats/complex-sample."""

    evidence = _finite_float(log_evidence, "log_evidence")
    count = _strict_int(sample_count, "sample_count", minimum=1)
    result = evidence / count
    if not isfinite(result):
        raise RuntimeError("per-sample score is non-finite")
    return result


@dataclass(frozen=True)
class IQObservationReceipt:
    """Canonical, hashable result of one independent likelihood evaluation."""

    reference_id: str
    sample_count: int
    integrated_i: float
    integrated_q: float
    within_window_residual: float
    component_log_likelihoods: tuple[float, ...]
    log_evidence: float
    posterior: tuple[float, ...]
    pairwise_llr: tuple[tuple[float, ...], ...]
    input_sha256: str
    raw_base_measure: str = RAW_BASE_MEASURE
    sigma_convention: str = SIGMA_CONVENTION
    integration_convention: str = INTEGRATION_CONVENTION

    def __post_init__(self) -> None:
        if self.reference_id != REFERENCE_ID:
            raise ValueError("receipt reference_id is immutable")
        if self.raw_base_measure != RAW_BASE_MEASURE:
            raise ValueError("receipt raw_base_measure is immutable")
        if self.sigma_convention != SIGMA_CONVENTION:
            raise ValueError("receipt sigma_convention is immutable")
        if self.integration_convention != INTEGRATION_CONVENTION:
            raise ValueError("receipt integration_convention is immutable")
        _strict_int(self.sample_count, "sample_count", minimum=1)
        for name in (
            "integrated_i",
            "integrated_q",
            "within_window_residual",
            "log_evidence",
        ):
            _finite_float(getattr(self, name), name)
        logs = _finite_vector(
            self.component_log_likelihoods,
            "component_log_likelihoods",
            minimum_length=2,
        )
        posterior = _finite_vector(
            self.posterior,
            "posterior",
            minimum_length=2,
        )
        if len(logs) != len(posterior):
            raise ValueError("receipt likelihood and posterior lengths differ")
        if (
            any(value < 0.0 or value > 1.0 for value in posterior)
            or abs(sum(posterior) - 1.0) > 5.0e-13
        ):
            raise ValueError("receipt posterior is not normalized")
        if len(self.pairwise_llr) != len(logs) or any(
            len(row) != len(logs) for row in self.pairwise_llr
        ):
            raise ValueError("receipt pairwise_llr must be square")
        for row in self.pairwise_llr:
            _finite_vector(row, "pairwise_llr row", minimum_length=len(logs))
        if (
            not isinstance(self.input_sha256, str)
            or len(self.input_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.input_sha256)
        ):
            raise ValueError("input_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "sample_count": self.sample_count,
            "integrated_i": self.integrated_i,
            "integrated_q": self.integrated_q,
            "within_window_residual": self.within_window_residual,
            "component_log_likelihoods": list(
                self.component_log_likelihoods
            ),
            "log_evidence": self.log_evidence,
            "posterior": list(self.posterior),
            "pairwise_llr": [list(row) for row in self.pairwise_llr],
            "input_sha256": self.input_sha256,
            "raw_base_measure": self.raw_base_measure,
            "sigma_convention": self.sigma_convention,
            "integration_convention": self.integration_convention,
        }

    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def evaluate_observation(
    iq_i: Sequence[object],
    iq_q: Sequence[object],
    *,
    priors: Sequence[object],
    centers: Sequence[Sequence[object]],
    sigma: object,
) -> IQObservationReceipt:
    """Evaluate and bind one raw IQ window."""

    i_values, q_values = _samples(iq_i, iq_q)
    means = _centers(centers)
    probabilities = _probabilities(priors, expected_length=len(means))
    scale = _positive_float(sigma, "sigma")
    integrated, residual = within_window_residual(i_values, q_values)
    component_logs = component_log_likelihoods(
        i_values,
        q_values,
        centers=means,
        sigma=scale,
    )
    decomposed = residual_decomposed_log_likelihoods(
        i_values,
        q_values,
        centers=means,
        sigma=scale,
    )
    if any(
        abs(direct - split) > 5.0e-12
        for direct, split in zip(component_logs, decomposed)
    ):
        raise RuntimeError("direct and residual likelihood paths disagree")
    evidence, posterior = evidence_and_posterior(
        component_logs,
        priors=probabilities,
    )
    llr = pairwise_log_likelihood_ratios(component_logs)
    input_payload = {
        "reference_id": REFERENCE_ID,
        "iq_i": list(i_values),
        "iq_q": list(q_values),
        "priors": list(probabilities),
        "centers": [list(center) for center in means],
        "sigma": scale,
    }
    input_digest = sha256(
        json.dumps(
            input_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return IQObservationReceipt(
        reference_id=REFERENCE_ID,
        sample_count=len(i_values),
        integrated_i=integrated[0],
        integrated_q=integrated[1],
        within_window_residual=residual,
        component_log_likelihoods=component_logs,
        log_evidence=evidence,
        posterior=posterior,
        pairwise_llr=llr,
        input_sha256=input_digest,
    )


__all__ = [
    "INTEGRATION_CONVENTION",
    "IQObservationReceipt",
    "RAW_BASE_MEASURE",
    "REFERENCE_ID",
    "SIGMA_CONVENTION",
    "affine_log_density_correction",
    "component_log_likelihoods",
    "evaluate_observation",
    "evidence_and_posterior",
    "integrated_marginal_cdf",
    "integrated_mean_gain_jacobian",
    "integrated_predictive_moments",
    "pairwise_log_likelihood_ratios",
    "per_complex_sample_log_score",
    "residual_decomposed_log_likelihoods",
    "within_window_residual",
]
