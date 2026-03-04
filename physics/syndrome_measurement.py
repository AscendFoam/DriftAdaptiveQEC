"""
Syndrome Measurement Module

Implements GKP syndrome measurement with realistic effects:
- Finite squeezing noise
- Measurement inefficiency
- Ancilla errors
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .gkp_state import LATTICE_CONST


@dataclass
class MeasurementConfig:
    """Configuration for syndrome measurement"""
    delta: float = 0.3  # GKP state finite energy parameter
    measurement_efficiency: float = 0.95  # Detector efficiency
    ancilla_error_rate: float = 0.01  # Probability of ancilla error
    add_shot_noise: bool = True  # Include shot noise


class SyndromeMeasurement:
    """
    Basic syndrome measurement for GKP codes

    The syndrome is the displacement error modulo the lattice constant.
    For an error displacement (e_q, e_p), the syndrome is:
    s_q = e_q mod √(2π) mapped to [-√(2π)/2, √(2π)/2]
    s_p = e_p mod √(2π) mapped to [-√(2π)/2, √(2π)/2]
    """

    def __init__(self, lattice: float = LATTICE_CONST):
        self.lattice = lattice

    def measure(self, displacement: np.ndarray) -> np.ndarray:
        """
        Ideal syndrome measurement

        Args:
            displacement: [dq, dp] displacement error

        Returns:
            syndrome: [sq, sp] measured syndrome
        """
        # Map to [-lattice/2, lattice/2]
        syndrome = np.mod(displacement + self.lattice / 2, self.lattice) - self.lattice / 2
        return syndrome

    def get_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Get correction displacement from syndrome

        For ideal measurement, correction = -syndrome
        """
        return -syndrome


class RealisticSyndromeMeasurement:
    """
    Realistic syndrome measurement including noise sources

    In real GKP error correction:
    1. An ancilla GKP state is prepared
    2. A SUM gate couples the data and ancilla
    3. Homodyne detection measures the ancilla
    4. The measurement has noise from finite squeezing

    The measurement noise scales with the GKP delta parameter:
    σ_meas ∝ Δ
    """

    def __init__(self, config: Optional[MeasurementConfig] = None):
        """
        Args:
            config: Measurement configuration
        """
        self.config = config or MeasurementConfig()
        self.lattice = LATTICE_CONST

        # Compute measurement noise variance
        self._compute_noise_variance()

    def _compute_noise_variance(self):
        """Compute measurement noise from configuration"""
        delta = self.config.delta
        eta = self.config.measurement_efficiency

        # Finite squeezing contributes measurement noise
        # Variance scales as Δ² for GKP states
        var_squeezing = delta ** 2

        # Measurement inefficiency adds vacuum noise
        # (1-η)/η times the shot noise variance (=1/2)
        var_inefficiency = (1 - eta) / (2 * eta) if eta > 0 else 1.0

        # Total measurement variance
        self.sigma_meas = np.sqrt(var_squeezing + var_inefficiency)

    def measure(self,
                true_displacement: np.ndarray,
                add_noise: bool = True) -> np.ndarray:
        """
        Simulate realistic syndrome measurement

        Args:
            true_displacement: [dq, dp] actual displacement error
            add_noise: Whether to add measurement noise

        Returns:
            syndrome: [sq, sp] measured syndrome (with noise)
        """
        # Ideal syndrome
        syndrome_ideal = np.mod(true_displacement + self.lattice / 2,
                                self.lattice) - self.lattice / 2

        if not add_noise:
            return syndrome_ideal

        # Add measurement noise
        measurement_noise = np.random.normal(0, self.sigma_meas, size=2)

        # Shot noise (detector noise)
        if self.config.add_shot_noise:
            shot_noise = np.random.normal(0, 0.1, size=2)
        else:
            shot_noise = 0

        syndrome_noisy = syndrome_ideal + measurement_noise + shot_noise

        # Ancilla errors (rare bit flips in measurement)
        if np.random.rand() < self.config.ancilla_error_rate:
            # Random offset by half lattice
            if np.random.rand() > 0.5:
                syndrome_noisy[0] += self.lattice / 2 * np.sign(np.random.randn())
            else:
                syndrome_noisy[1] += self.lattice / 2 * np.sign(np.random.randn())

        return syndrome_noisy

    def get_correction(self,
                       syndrome: np.ndarray,
                       gain: float = 1.0) -> np.ndarray:
        """
        Get correction displacement from syndrome

        Args:
            syndrome: Measured syndrome [sq, sp]
            gain: Correction gain (< 1 for noisy measurements)

        Returns:
            correction: Correction displacement [dq, dp]
        """
        return -gain * syndrome

    def get_optimal_gain(self) -> float:
        """
        Compute optimal correction gain

        For noisy measurements, the optimal gain is:
        g* = σ_signal² / (σ_signal² + σ_noise²)

        where σ_signal² is the variance of the true syndrome
        and σ_noise² is the measurement noise variance.
        """
        # Signal variance: uniform distribution in [-λ/2, λ/2]
        var_signal = (self.lattice / 2) ** 2 / 3

        # Noise variance
        var_noise = self.sigma_meas ** 2

        return var_signal / (var_signal + var_noise)

    def get_measurement_covariance(self) -> np.ndarray:
        """Get measurement noise covariance matrix"""
        var = self.sigma_meas ** 2
        if self.config.add_shot_noise:
            var += 0.01  # Shot noise variance
        return np.diag([var, var])

    def update_delta(self, delta: float):
        """Update GKP delta parameter"""
        self.config.delta = delta
        self._compute_noise_variance()


class AdaptiveSyndromeMeasurement(RealisticSyndromeMeasurement):
    """
    Syndrome measurement with adaptive gain

    The gain is adjusted based on estimated noise level.
    """

    def __init__(self, config: Optional[MeasurementConfig] = None):
        super().__init__(config)
        self.gain = self.get_optimal_gain()

    def measure_and_correct(self,
                            true_displacement: np.ndarray,
                            add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measure syndrome and compute correction in one step

        Returns:
            syndrome: Measured syndrome
            correction: Computed correction with optimal gain
        """
        syndrome = self.measure(true_displacement, add_noise)
        correction = self.get_correction(syndrome, self.gain)
        return syndrome, correction

    def update_gain(self, new_gain: float):
        """Update the correction gain"""
        self.gain = np.clip(new_gain, 0.1, 1.5)  # Safety bounds

    def adapt_to_noise(self, estimated_sigma: float):
        """
        Adapt gain based on estimated noise level

        Args:
            estimated_sigma: Estimated total noise standard deviation
        """
        # Update effective measurement noise
        self.sigma_meas = np.sqrt(self.config.delta**2 + estimated_sigma**2)
        self.gain = self.get_optimal_gain()


def simulate_measurement_statistics(n_samples: int = 10000,
                                    true_sigma: float = 0.3,
                                    delta: float = 0.3) -> dict:
    """
    Simulate measurement statistics for analysis

    Returns statistics on syndrome measurements including
    mean, variance, and correlation.
    """
    config = MeasurementConfig(delta=delta)
    measurement = RealisticSyndromeMeasurement(config)

    syndromes = []
    for _ in range(n_samples):
        # Random displacement error
        error = np.random.normal(0, true_sigma, size=2)
        syndrome = measurement.measure(error, add_noise=True)
        syndromes.append(syndrome)

    syndromes = np.array(syndromes)

    return {
        'mean_q': np.mean(syndromes[:, 0]),
        'mean_p': np.mean(syndromes[:, 1]),
        'std_q': np.std(syndromes[:, 0]),
        'std_p': np.std(syndromes[:, 1]),
        'correlation': np.corrcoef(syndromes[:, 0], syndromes[:, 1])[0, 1],
        'measurement_sigma': measurement.sigma_meas,
        'optimal_gain': measurement.get_optimal_gain(),
    }
