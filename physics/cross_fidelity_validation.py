"""T2.3.3 cross-fidelity validation across four explicitly bounded lanes.

The common comparison is a one-axis residual-parity response under one frozen
noise contract.  A square-code independence projection converts that response
to Pauli-twirled ``LER``/``F_avg`` only where stated.  Model-native occupancy
and protocol metrics remain separate; in particular, SBS code survival is not
silently equated with a Gaussian central-cell probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, pi, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from .constants import LATTICE_CONST
from ._shared.numerics import hermite_functions as _hermite_functions
from ._shared.validation import finite_positive as _positive
from ._cross_fidelity.reporting import (
    _error_attributions,
    _validation_checks,
    _write_cross_fidelity_validation,
)
from .finite_energy_gkp import damped_projector_state
from .finite_squeezing_noise import (
    FiniteSqueezingNoiseConfig,
    sample_finite_squeezing_noise,
)
from .fock_density_model import FiniteCutoffFockModel
from .fock_sbs_cycle import (
    SBSFockCycleConfig,
    SBSFockOneRoundSimulator,
    logical_density,
)
from .logical_channel import (
    finite_energy_parity_response_1d,
    parity_confusion_from_response,
)
from .noise_transfer_surrogate import (
    GKPNoiseTransferSurrogate,
    NoiseTransferConfig,
    NoiseTransferState,
    projector_delta_from_squeezing_db,
    squeezing_db_to_peak_variance,
)
from .quadrature_conventions import (
    CANONICAL_LOGICAL_CELL_SPACING,
    DECODER_STANDARDIZATION_SCALE,
    convert_axis_sigma,
)


RealMatrix = NDArray[np.float64]
Region = Literal["low_squeezing_clipping", "transition", "high_squeezing"]
FockCoordinateContract = Literal[
    "canonical_from_decoder",
    "legacy_ambiguous_operational_fourier",
]

CROSS_FIDELITY_SCOPE = (
    "four-lane directional cross-fidelity validation with canonical axis-resolved "
    "q/p Fock metrics, model-native occupancy separation and explicit legacy/cutoff "
    "failure attribution"
)
DEFAULT_DB_GRID = (3.0, 5.0, 8.0, 10.0, 12.0)


@dataclass(frozen=True)
class CrossFidelityConfig:
    squeezing_db: tuple[float, ...] = DEFAULT_DB_GRID
    channel_sigma: float = 0.18
    measurement_sigma: float = 0.06
    fock_cutoff: int = 48
    fock_protocol_cutoff: int = 42
    fock_points_per_cell: int = 512
    projection_grid_points: int = 4097
    syndrome_points: int = 1024
    effective_samples: int = 200_000
    seed: int = 2026071433
    scope: str = CROSS_FIDELITY_SCOPE

    def __post_init__(self) -> None:
        grid = tuple(float(item) for item in self.squeezing_db)
        if grid != DEFAULT_DB_GRID:
            raise ValueError(
                "squeezing_db must preserve the registered 3/5/8/10/12 dB grid"
            )
        object.__setattr__(self, "squeezing_db", grid)
        object.__setattr__(
            self, "channel_sigma", _positive(self.channel_sigma, "channel_sigma")
        )
        object.__setattr__(
            self,
            "measurement_sigma",
            _positive(self.measurement_sigma, "measurement_sigma"),
        )
        for name, lower, upper in (
            ("fock_cutoff", 24, 48),
            ("fock_protocol_cutoff", 24, 48),
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if not lower <= int(value) <= upper:
                raise ValueError(f"{name} must lie in [{lower}, {upper}]")
            object.__setattr__(self, name, int(value))
        if (
            not isinstance(self.fock_points_per_cell, (int, np.integer))
            or int(self.fock_points_per_cell) < 256
            or int(self.fock_points_per_cell) % 2 != 0
        ):
            raise ValueError("fock_points_per_cell must be an even integer >= 256")
        object.__setattr__(
            self, "fock_points_per_cell", int(self.fock_points_per_cell)
        )
        if (
            not isinstance(self.projection_grid_points, (int, np.integer))
            or int(self.projection_grid_points) < 2049
            or int(self.projection_grid_points) % 2 == 0
        ):
            raise ValueError(
                "projection_grid_points must be an odd integer >= 2049"
            )
        object.__setattr__(
            self, "projection_grid_points", int(self.projection_grid_points)
        )
        if (
            not isinstance(self.syndrome_points, (int, np.integer))
            or not 512 <= int(self.syndrome_points) <= 65_536
        ):
            raise ValueError("syndrome_points must be an integer in [512, 65536]")
        object.__setattr__(self, "syndrome_points", int(self.syndrome_points))
        if (
            not isinstance(self.effective_samples, (int, np.integer))
            or int(self.effective_samples) < 100_000
        ):
            raise ValueError("effective_samples must be an integer >= 100000")
        object.__setattr__(self, "effective_samples", int(self.effective_samples))
        if not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "seed", int(self.seed))
        if self.scope != CROSS_FIDELITY_SCOPE:
            raise ValueError("scope must preserve the fail-closed comparison boundary")


@dataclass(frozen=True)
class PauliTrendMetrics:
    q_axis_ler: float
    p_axis_ler: float
    logical_error_rate: float
    correct_coset_occupancy: float
    average_fidelity: float
    construction: str


def independent_axis_pauli_metrics(
    q_axis_ler: float,
    p_axis_ler: float,
    *,
    construction: str,
) -> PauliTrendMetrics:
    q = float(q_axis_ler)
    p = float(p_axis_ler)
    if (
        not isfinite(q)
        or not isfinite(p)
        or not 0.0 <= q <= 1.0
        or not 0.0 <= p <= 1.0
    ):
        raise ValueError("axis logical-error rates must lie in [0, 1]")
    if not isinstance(construction, str) or not construction.strip():
        raise ValueError("construction must be a non-empty string")
    identity = (1.0 - q) * (1.0 - p)
    logical_error = 1.0 - identity
    return PauliTrendMetrics(
        q_axis_ler=q,
        p_axis_ler=p,
        logical_error_rate=logical_error,
        correct_coset_occupancy=identity,
        average_fidelity=(2.0 * identity + 1.0) / 3.0,
        construction=construction.strip(),
    )


@dataclass(frozen=True)
class FockFoldedResponse:
    logical_label: str
    quadrature: Literal["q", "p"]
    map_error_probability: float
    captured_probability: float
    reconstructed_mass: float
    cutoff: int
    alias_radius: int
    source_coordinate_scale: float
    domain_spacing: float
    displacement_sigma_input: float
    displacement_sigma_canonical: float
    coordinate_contract: FockCoordinateContract


def fock_folded_map_response(
    logical_label: str,
    projector_delta: float,
    displacement_sigma: float,
    *,
    quadrature: Literal["q", "p"] = "q",
    cutoff: int = 48,
    points_per_cell: int = 512,
    projection_grid_points: int = 4097,
    source_coordinate_scale: float | None = None,
    domain_spacing: float | None = None,
    coordinate_contract: FockCoordinateContract = "canonical_from_decoder",
) -> FockFoldedResponse:
    if logical_label not in {"0", "1", "+", "-"}:
        raise ValueError("logical_label must be 0/1/+/-")
    if quadrature not in {"q", "p"}:
        raise ValueError("quadrature must be q or p")
    if quadrature == "q" and logical_label not in {"0", "1"}:
        raise ValueError("q response requires logical 0/1")
    if quadrature == "p" and logical_label not in {"+", "-"}:
        raise ValueError("p response requires logical +/-")
    delta = _positive(projector_delta, "projector_delta")
    sigma_input = float(displacement_sigma)
    if not isfinite(sigma_input) or sigma_input < 0.0:
        raise ValueError("displacement_sigma must be finite and non-negative")
    if not isinstance(cutoff, (int, np.integer)) or not 12 <= int(cutoff) <= 48:
        raise ValueError("cutoff must be an integer in [12, 48]")
    if (
        not isinstance(points_per_cell, (int, np.integer))
        or int(points_per_cell) < 128
        or int(points_per_cell) % 2 != 0
    ):
        raise ValueError("points_per_cell must be an even integer >= 128")
    if (
        not isinstance(projection_grid_points, (int, np.integer))
        or int(projection_grid_points) < 1025
        or int(projection_grid_points) % 2 == 0
    ):
        raise ValueError("projection_grid_points must be an odd integer >= 1025")
    if coordinate_contract == "canonical_from_decoder":
        expected_scale = DECODER_STANDARDIZATION_SCALE
        expected_spacing = CANONICAL_LOGICAL_CELL_SPACING
        scale = expected_scale if source_coordinate_scale is None else _positive(
            source_coordinate_scale, "source_coordinate_scale"
        )
        spacing = expected_spacing if domain_spacing is None else _positive(
            domain_spacing, "domain_spacing"
        )
        if not np.isclose(scale, expected_scale, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "canonical_from_decoder requires source_coordinate_scale=sqrt(2)"
            )
        if not np.isclose(spacing, expected_spacing, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "canonical_from_decoder requires domain_spacing=sqrt(pi)"
            )
        sigma = convert_axis_sigma(
            sigma_input,
            "decoder_standardized",
            "canonical_fock",
            quadrature,
        )
    elif coordinate_contract == "legacy_ambiguous_operational_fourier":
        expected_scale = 1.0
        expected_spacing = LATTICE_CONST
        scale = expected_scale if source_coordinate_scale is None else _positive(
            source_coordinate_scale, "source_coordinate_scale"
        )
        spacing = expected_spacing if domain_spacing is None else _positive(
            domain_spacing, "domain_spacing"
        )
        if not np.isclose(scale, expected_scale, rtol=0.0, atol=1.0e-14):
            raise ValueError("legacy audit freezes source_coordinate_scale=1")
        if not np.isclose(spacing, expected_spacing, rtol=0.0, atol=1.0e-14):
            raise ValueError("legacy audit freezes domain_spacing=sqrt(2*pi)")
        sigma = sigma_input
    else:
        raise ValueError("unknown Fock coordinate contract")
    model = FiniteCutoffFockModel(int(cutoff))
    if coordinate_contract == "canonical_from_decoder":
        preparation = model.prepare_damped_projector_gkp(
            logical_label,
            delta,
            grid_points=int(projection_grid_points),
            source_coordinate_scale=scale,
        )
    else:
        preparation = model.project_finite_energy_gkp(
            damped_projector_state(logical_label, delta),
            grid_points=int(projection_grid_points),
            source_coordinate_scale=scale,
        )
    extent = max(preparation.q_extent, sqrt(2.0 * int(cutoff)) + 6.0)
    alias_radius = int(ceil((extent + 8.0 * sigma) / spacing)) + 1
    aliases = np.arange(-alias_radius, alias_radius + 1, dtype=np.int64)
    points = int(points_per_cell)
    residual = (-0.5 + (np.arange(points) + 0.5) / points) * spacing
    coordinate = (aliases[:, np.newaxis] * spacing + residual).reshape(-1)
    functions = _hermite_functions(coordinate, int(cutoff))
    coefficients = preparation.coefficients / np.linalg.norm(preparation.coefficients)
    if quadrature == "p":
        coefficients = coefficients * ((-1.0j) ** np.arange(int(cutoff)))
    density = np.abs(coefficients @ functions) ** 2
    step = spacing / points
    if sigma > 0.0:
        density = gaussian_filter1d(
            density,
            sigma / step,
            mode="constant",
            truncate=8.0,
        )
    density = density.reshape(aliases.size, points)
    joint = np.stack(
        (
            np.sum(density[np.mod(aliases, 2) == 0], axis=0),
            np.sum(density[np.mod(aliases, 2) == 1], axis=0),
        )
    )
    reference_parity = 0 if logical_label in {"0", "+"} else 1
    if reference_parity == 1:
        joint = joint[::-1]
    mass = float(np.sum(joint) * step)
    if not isfinite(mass) or mass <= 0.0:
        raise RuntimeError("Fock folded response has invalid mass")
    error = float(np.sum(np.minimum(joint[0], joint[1])) * step / mass)
    return FockFoldedResponse(
        logical_label=logical_label,
        quadrature=quadrature,
        map_error_probability=error,
        captured_probability=preparation.captured_probability,
        reconstructed_mass=mass,
        cutoff=int(cutoff),
        alias_radius=alias_radius,
        source_coordinate_scale=scale,
        domain_spacing=spacing,
        displacement_sigma_input=sigma_input,
        displacement_sigma_canonical=sigma,
        coordinate_contract=coordinate_contract,
    )


@dataclass(frozen=True)
class FockProtocolMetrics:
    average_conditional_fidelity: float
    average_code_survival: float
    average_code_weighted_fidelity: float
    cutoff: int
    scope: str = "finite_cutoff_completed_analytic_sbs_native_metrics"


def _fock_protocol_metrics(
    projector_delta: float,
    *,
    cutoff: int,
    projection_grid_points: int,
) -> FockProtocolMetrics:
    simulator = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=cutoff,
            projector_delta=projector_delta,
            grid_points=projection_grid_points,
            readout_confusion=((1.0, 0.0), (0.0, 1.0)),
        )
    )
    fidelities = []
    survivals = []
    weighted = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        target = logical_density(label)
        projection = simulator.run_exact_cycle(
            simulator.initialize(label)
        ).unconditional_projection
        fidelity = float(
            np.trace(projection.frame_corrected_logical_density @ target).real
        )
        fidelities.append(fidelity)
        survivals.append(projection.code_survival_probability)
        weighted.append(projection.code_survival_probability * fidelity)
    return FockProtocolMetrics(
        average_conditional_fidelity=float(np.mean(fidelities)),
        average_code_survival=float(np.mean(survivals)),
        average_code_weighted_fidelity=float(np.mean(weighted)),
        cutoff=cutoff,
    )


@dataclass(frozen=True)
class FockLaneMetrics:
    q_responses: tuple[FockFoldedResponse, FockFoldedResponse]
    canonical_p_responses: tuple[FockFoldedResponse, FockFoldedResponse]
    legacy_operational_p_fourier_audit: tuple[FockFoldedResponse, FockFoldedResponse]
    two_axis_pauli_metrics: PauliTrendMetrics
    minimum_captured_probability: float
    maximum_reconstructed_mass_error: float
    p_minus_q_ler_gap: float
    legacy_p_minus_q_ler_gap: float
    protocol: FockProtocolMetrics
    common_metric_scope: str = (
        "canonical q/p Fock responses converted axis-by-axis from decoder-standardized inputs; "
        "legacy non-symplectic Fourier audit retained separately as negative provenance"
    )


@dataclass(frozen=True)
class EffectiveLaneMetrics:
    pauli: PauliTrendMetrics
    central_domain_occupancy: float
    observed_covariance_trace: float
    envelope_variance_trace: float
    samples: int
    maximum_axis_standard_error: float
    scope: str = "decomposed_stochastic_finite_squeezing_effective_lane"


@dataclass(frozen=True)
class NoiseTransferLaneMetrics:
    pauli: PauliTrendMetrics
    central_domain_occupancy: float
    minimum_clipping_ratio: float
    validity: str
    decision_variance: float
    scope: str = "localized_Gaussian_noise_transfer_lane"


@dataclass(frozen=True)
class SyndromeLaneMetrics:
    logical_zero_axis_ler: float
    logical_one_axis_ler: float
    square_symmetry_projection: PauliTrendMetrics
    minimum_captured_mass: float
    scope: str = "direct_finite_energy_syndrome_density_MAP_lane"


@dataclass(frozen=True)
class CrossFidelityPoint:
    squeezing_db: float
    projector_delta: float
    peak_variance: float
    external_displacement_sigma: float
    region: Region
    fock: FockLaneMetrics
    effective: EffectiveLaneMetrics
    noise_transfer: NoiseTransferLaneMetrics
    syndrome: SyndromeLaneMetrics


def _region(db: float) -> Region:
    if db <= 5.0:
        return "low_squeezing_clipping"
    if db >= 10.0:
        return "high_squeezing"
    return "transition"


def _fock_lane(
    db: float,
    delta: float,
    external_sigma: float,
    config: CrossFidelityConfig,
) -> FockLaneMetrics:
    q_responses = tuple(
        fock_folded_map_response(
            label,
            delta,
            external_sigma,
            quadrature="q",
            cutoff=config.fock_cutoff,
            points_per_cell=config.fock_points_per_cell,
            projection_grid_points=config.projection_grid_points,
        )
        for label in ("0", "1")
    )
    p_responses = tuple(
        fock_folded_map_response(
            label,
            delta,
            external_sigma,
            quadrature="p",
            cutoff=config.fock_cutoff,
            points_per_cell=config.fock_points_per_cell,
            projection_grid_points=config.projection_grid_points,
        )
        for label in ("+", "-")
    )
    legacy_p_responses = tuple(
        fock_folded_map_response(
            label,
            delta,
            external_sigma,
            quadrature="p",
            cutoff=config.fock_cutoff,
            points_per_cell=config.fock_points_per_cell,
            projection_grid_points=config.projection_grid_points,
            coordinate_contract="legacy_ambiguous_operational_fourier",
        )
        for label in ("+", "-")
    )
    q_error = float(np.mean([item.map_error_probability for item in q_responses]))
    p_error = float(np.mean([item.map_error_probability for item in p_responses]))
    legacy_p_audit = float(
        np.mean([item.map_error_probability for item in legacy_p_responses])
    )
    all_responses = q_responses + p_responses
    return FockLaneMetrics(
        q_responses=(q_responses[0], q_responses[1]),
        canonical_p_responses=(p_responses[0], p_responses[1]),
        legacy_operational_p_fourier_audit=(
            legacy_p_responses[0],
            legacy_p_responses[1],
        ),
        two_axis_pauli_metrics=independent_axis_pauli_metrics(
            q_error,
            p_error,
            construction=(
                "independent canonical q/p folded responses after explicit decoder-to-canonical "
                "axis conversion; no coherent joint-axis correlation claim"
            ),
        ),
        minimum_captured_probability=min(
            item.captured_probability for item in all_responses
        ),
        maximum_reconstructed_mass_error=max(
            abs(item.reconstructed_mass - 1.0) for item in all_responses
        ),
        p_minus_q_ler_gap=p_error - q_error,
        legacy_p_minus_q_ler_gap=legacy_p_audit - q_error,
        protocol=_fock_protocol_metrics(
            delta,
            cutoff=config.fock_protocol_cutoff,
            projection_grid_points=config.projection_grid_points,
        ),
    )


def _effective_lane(
    db: float,
    delta: float,
    config: CrossFidelityConfig,
) -> EffectiveLaneMetrics:
    channel_variance = config.channel_sigma**2
    measurement_variance = config.measurement_sigma**2
    batch = sample_finite_squeezing_noise(
        FiniteSqueezingNoiseConfig(
            channel_covariance=(
                (channel_variance, 0.0),
                (0.0, channel_variance),
            ),
            data_delta=(delta, delta),
            ancilla_delta=(delta, delta),
            measurement_covariance=(
                (measurement_variance, 0.0),
                (0.0, measurement_variance),
            ),
            include_envelope=True,
            samples=config.effective_samples,
            seed=config.seed + int(round(10.0 * db)),
        )
    )
    parity = batch.logical_parity
    q_error = float(np.mean(parity[:, 0] != 0))
    p_error = float(np.mean(parity[:, 1] != 0))
    aliases = np.floor(batch.corrected_residual / LATTICE_CONST + 0.5).astype(
        np.int64
    )
    central = float(np.mean(np.all(aliases == 0, axis=1)))
    maximum_se = max(
        sqrt(max(value * (1.0 - value), 1.0e-15) / config.effective_samples)
        for value in (q_error, p_error)
    )
    return EffectiveLaneMetrics(
        pauli=independent_axis_pauli_metrics(
            q_error,
            p_error,
            construction="empirical independent-axis Pauli aggregation of effective residual parities",
        ),
        central_domain_occupancy=central,
        observed_covariance_trace=float(np.trace(batch.budget.observed_total)),
        envelope_variance_trace=float(
            np.trace(batch.budget.finite_energy_envelope)
        ),
        samples=config.effective_samples,
        maximum_axis_standard_error=maximum_se,
    )


def _noise_transfer_lane(
    peak_variance: float,
    config: CrossFidelityConfig,
) -> NoiseTransferLaneMetrics:
    channel_variance = config.channel_sigma**2
    measurement_variance = config.measurement_sigma**2
    result = GKPNoiseTransferSurrogate(
        NoiseTransferConfig(
            resource_covariance=(
                (peak_variance + measurement_variance, 0.0),
                (0.0, peak_variance + measurement_variance),
            ),
            loss_transmissivity=1.0,
            measurement_efficiency=1.0,
        )
    ).propagate(
        NoiseTransferState(
            fluctuation_covariance=(
                (channel_variance + peak_variance, 0.0),
                (0.0, channel_variance + peak_variance),
            )
        )
    )
    q = result.logical_jump.q_odd_probability
    p = result.logical_jump.p_odd_probability
    diagnostics = [item.alias_statistics for item in result.axis_diagnostics]
    return NoiseTransferLaneMetrics(
        pauli=independent_axis_pauli_metrics(
            q,
            p,
            construction="exact product law for diagonal Gaussian decision covariance",
        ),
        central_domain_occupancy=float(
            diagnostics[0].central_probability * diagnostics[1].central_probability
        ),
        minimum_clipping_ratio=min(item.clipping_ratio for item in diagnostics),
        validity=result.validity,
        decision_variance=float(result.decision_covariance[0][0]),
    )


def _syndrome_lane(
    delta: float,
    external_sigma: float,
    config: CrossFidelityConfig,
) -> SyndromeLaneMetrics:
    responses = []
    errors = []
    for label in ("0", "1"):
        response = finite_energy_parity_response_1d(
            damped_projector_state(label, delta),
            displacement_sigma=external_sigma,
            points=config.syndrome_points,
        )
        responses.append(response)
        errors.append(parity_confusion_from_response(response).error_probability)
    axis_error = float(np.mean(errors))
    return SyndromeLaneMetrics(
        logical_zero_axis_ler=float(errors[0]),
        logical_one_axis_ler=float(errors[1]),
        square_symmetry_projection=independent_axis_pauli_metrics(
            axis_error,
            axis_error,
            construction="direct state-density MAP response copied by square-code q/p role symmetry",
        ),
        minimum_captured_mass=min(item.captured_mass for item in responses),
    )


@dataclass(frozen=True)
class FockCutoffPoint:
    cutoff: int
    q_axis_ler: float
    minimum_captured_probability: float
    maximum_reconstructed_mass_error: float


def _twelve_db_cutoff_sweep(
    config: CrossFidelityConfig,
) -> tuple[FockCutoffPoint, ...]:
    db = 12.0
    delta = projector_delta_from_squeezing_db(db)
    peak = squeezing_db_to_peak_variance(
        db, coordinate_chart="decoder_standardized"
    )
    external = sqrt(
        config.channel_sigma**2 + config.measurement_sigma**2 + peak
    )
    points = []
    for cutoff in (24, 30, 36, 42, 48):
        responses = [
            fock_folded_map_response(
                label,
                delta,
                external,
                cutoff=cutoff,
                points_per_cell=config.fock_points_per_cell,
                projection_grid_points=config.projection_grid_points,
            )
            for label in ("0", "1")
        ]
        points.append(
            FockCutoffPoint(
                cutoff=cutoff,
                q_axis_ler=float(
                    np.mean([item.map_error_probability for item in responses])
                ),
                minimum_captured_probability=min(
                    item.captured_probability for item in responses
                ),
                maximum_reconstructed_mass_error=max(
                    abs(item.reconstructed_mass - 1.0) for item in responses
                ),
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class ErrorAttribution:
    attribution_id: str
    region: str
    observation: str
    primary_cause: str
    reporting_consequence: str


@dataclass(frozen=True)
class CrossFidelityValidationResult:
    config: CrossFidelityConfig
    points: tuple[CrossFidelityPoint, ...]
    twelve_db_fock_cutoff_sweep: tuple[FockCutoffPoint, ...]
    maximum_high_squeezing_fock_syndrome_q_ler_gap: float
    maximum_high_squeezing_noise_syndrome_q_ler_gap: float
    maximum_high_squeezing_effective_noise_z_score: float
    maximum_high_squeezing_canonical_fock_qp_ler_gap: float
    minimum_high_squeezing_legacy_p_minus_q_ler_gap: float
    low_squeezing_noise_syndrome_q_ler_gap: float
    attributions: tuple[ErrorAttribution, ...]
    checks: dict[str, bool]
    scope: str = CROSS_FIDELITY_SCOPE

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _strictly_decreasing(values: list[float]) -> bool:
    return all(values[index] > values[index + 1] for index in range(len(values) - 1))


def _strictly_increasing(values: list[float]) -> bool:
    return all(values[index] < values[index + 1] for index in range(len(values) - 1))


def evaluate_cross_fidelity_point(
    squeezing_db: float,
    config: CrossFidelityConfig | None = None,
) -> CrossFidelityPoint:
    """Evaluate one explicitly supplied point without relaxing the T2.3.3 grid.

    ``CrossFidelityConfig.squeezing_db`` remains frozen to the original
    calibration grid.  This point-level entry point exists so later tasks can
    preregister disjoint holdout points while reusing exactly the same four
    model lanes and numerical settings.
    """

    actual = CrossFidelityConfig() if config is None else config
    if not isinstance(actual, CrossFidelityConfig):
        raise TypeError("config must be a CrossFidelityConfig or None")
    db = float(squeezing_db)
    if not isfinite(db) or not 0.0 < db <= 20.0:
        raise ValueError("squeezing_db must be finite and lie in (0, 20]")
    delta = projector_delta_from_squeezing_db(db)
    peak = squeezing_db_to_peak_variance(
        db, coordinate_chart="decoder_standardized"
    )
    external = sqrt(
        actual.channel_sigma**2 + actual.measurement_sigma**2 + peak
    )
    return CrossFidelityPoint(
        squeezing_db=db,
        projector_delta=delta,
        peak_variance=peak,
        external_displacement_sigma=external,
        region=_region(db),
        fock=_fock_lane(db, delta, external, actual),
        effective=_effective_lane(db, delta, actual),
        noise_transfer=_noise_transfer_lane(peak, actual),
        syndrome=_syndrome_lane(delta, external, actual),
    )


@dataclass(frozen=True)
class _ComparisonGaps:
    high: list[CrossFidelityPoint]
    low: CrossFidelityPoint
    fock_syndrome: float
    noise_syndrome: float
    effective_noise_z: float
    low_noise_syndrome: float


@dataclass(frozen=True)
class _DirectionalTrends:
    common_lanes_consistent: bool
    protocol_survival: list[float]
    protocol_weighted: list[float]
    effective_occupancy: list[float]
    noise_occupancy: list[float]
    cutoff_ler: list[float]
    cutoff_capture: list[float]


def _comparison_gaps(points: list[CrossFidelityPoint]) -> _ComparisonGaps:
    high = [item for item in points if item.region == "high_squeezing"]
    fock_syndrome_gap = max(
        abs(
            item.fock.two_axis_pauli_metrics.q_axis_ler
            - item.syndrome.square_symmetry_projection.q_axis_ler
        )
        for item in high
    )
    noise_syndrome_gap = max(
        abs(
            item.noise_transfer.pauli.q_axis_ler
            - item.syndrome.square_symmetry_projection.q_axis_ler
        )
        for item in high
    )
    z_scores = []
    for item in high:
        prediction = item.noise_transfer.pauli.q_axis_ler
        standard_error = sqrt(
            max(prediction * (1.0 - prediction), 1.0e-15)
            / item.effective.samples
        )
        z_scores.extend(
            [
                abs(item.effective.pauli.q_axis_ler - prediction) / standard_error,
                abs(item.effective.pauli.p_axis_ler - prediction) / standard_error,
            ]
        )
    max_z = max(z_scores)
    low = points[0]
    low_gap = abs(
        low.noise_transfer.pauli.q_axis_ler
        - low.syndrome.square_symmetry_projection.q_axis_ler
    )
    return _ComparisonGaps(
        high=high,
        low=low,
        fock_syndrome=fock_syndrome_gap,
        noise_syndrome=noise_syndrome_gap,
        effective_noise_z=max_z,
        low_noise_syndrome=low_gap,
    )


def _directional_trends(
    points: list[CrossFidelityPoint],
    cutoff_sweep: tuple[FockCutoffPoint, ...],
) -> _DirectionalTrends:
    common_lanes = {
        "fock_canonical_two_axis": [
            item.fock.two_axis_pauli_metrics for item in points
        ],
        "effective": [item.effective.pauli for item in points],
        "noise_transfer": [item.noise_transfer.pauli for item in points],
        "syndrome": [
            item.syndrome.square_symmetry_projection for item in points
        ],
    }
    directional = all(
        _strictly_decreasing([metric.logical_error_rate for metric in metrics])
        and _strictly_increasing(
            [metric.correct_coset_occupancy for metric in metrics]
        )
        and _strictly_increasing([metric.average_fidelity for metric in metrics])
        for metrics in common_lanes.values()
    )
    protocol_survival = [item.fock.protocol.average_code_survival for item in points]
    protocol_weighted = [
        item.fock.protocol.average_code_weighted_fidelity for item in points
    ]
    effective_occupancy = [item.effective.central_domain_occupancy for item in points]
    noise_occupancy = [item.noise_transfer.central_domain_occupancy for item in points]
    cutoff_ler = [item.q_axis_ler for item in cutoff_sweep]
    cutoff_capture = [item.minimum_captured_probability for item in cutoff_sweep]
    return _DirectionalTrends(
        common_lanes_consistent=directional,
        protocol_survival=protocol_survival,
        protocol_weighted=protocol_weighted,
        effective_occupancy=effective_occupancy,
        noise_occupancy=noise_occupancy,
        cutoff_ler=cutoff_ler,
        cutoff_capture=cutoff_capture,
    )


def _axis_comparison_gaps(
    high: list[CrossFidelityPoint],
) -> tuple[float, float]:
    canonical_qp_gap = max(abs(item.fock.p_minus_q_ler_gap) for item in high)
    legacy_p_audit_gap = min(
        item.fock.legacy_p_minus_q_ler_gap for item in high
    )
    return canonical_qp_gap, legacy_p_audit_gap


def run_cross_fidelity_validation(
    config: CrossFidelityConfig | None = None,
) -> CrossFidelityValidationResult:
    actual = CrossFidelityConfig() if config is None else config
    if not isinstance(actual, CrossFidelityConfig):
        raise TypeError("config must be a CrossFidelityConfig or None")
    points = [evaluate_cross_fidelity_point(db, actual) for db in actual.squeezing_db]
    cutoff_sweep = _twelve_db_cutoff_sweep(actual)
    comparison = _comparison_gaps(points)
    trends = _directional_trends(points, cutoff_sweep)
    canonical_qp_gap, legacy_p_audit_gap = _axis_comparison_gaps(comparison.high)
    attributions = _error_attributions(
        comparison, trends, canonical_qp_gap, legacy_p_audit_gap
    )
    checks = _validation_checks(
        points,
        comparison,
        trends,
        canonical_qp_gap,
        legacy_p_audit_gap,
        attributions,
    )
    return CrossFidelityValidationResult(
        config=actual,
        points=tuple(points),
        twelve_db_fock_cutoff_sweep=cutoff_sweep,
        maximum_high_squeezing_fock_syndrome_q_ler_gap=comparison.fock_syndrome,
        maximum_high_squeezing_noise_syndrome_q_ler_gap=comparison.noise_syndrome,
        maximum_high_squeezing_effective_noise_z_score=comparison.effective_noise_z,
        maximum_high_squeezing_canonical_fock_qp_ler_gap=canonical_qp_gap,
        minimum_high_squeezing_legacy_p_minus_q_ler_gap=legacy_p_audit_gap,
        low_squeezing_noise_syndrome_q_ler_gap=comparison.low_noise_syndrome,
        attributions=attributions,
        checks=checks,
    )


def write_cross_fidelity_validation(
    result: CrossFidelityValidationResult,
    output: str | Path,
) -> Path:
    return _write_cross_fidelity_validation(result, output)


if __name__ == "__main__":
    from ._cross_fidelity.reporting import main

    raise SystemExit(main())


__all__ = [
    "CROSS_FIDELITY_SCOPE",
    "DEFAULT_DB_GRID",
    "CrossFidelityConfig",
    "PauliTrendMetrics",
    "FockFoldedResponse",
    "FockProtocolMetrics",
    "FockLaneMetrics",
    "EffectiveLaneMetrics",
    "NoiseTransferLaneMetrics",
    "SyndromeLaneMetrics",
    "CrossFidelityPoint",
    "FockCutoffPoint",
    "ErrorAttribution",
    "CrossFidelityValidationResult",
    "independent_axis_pauli_metrics",
    "fock_folded_map_response",
    "run_cross_fidelity_validation",
    "write_cross_fidelity_validation",
]
