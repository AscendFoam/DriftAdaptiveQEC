"""
Logical Error Tracking Module

Tracks logical errors over multiple QEC rounds.
GKP logical errors occur when:
- Accumulated q displacement exceeds ±√(2π)/2 → Logical X error
- Accumulated p displacement exceeds ±√(2π)/2 → Logical Z error
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from .gkp_state import LATTICE_CONST


@dataclass
class LogicalErrorEvent:
    """Record of a logical error event"""
    timestep: int
    error_type: str  # 'X', 'Z', or 'Y'
    accumulated_q: float
    accumulated_p: float


class LogicalErrorTracker:
    """
    Tracks logical errors in GKP error correction

    GKP codes have logical Pauli operators:
    - X_L: Displacement by √(2π) in q
    - Z_L: Displacement by √(2π) in p

    A logical error occurs when the accumulated uncorrected
    displacement crosses the decision boundary at ±√(2π)/2.
    """

    def __init__(self):
        self.lattice = LATTICE_CONST
        self.reset()

    def reset(self):
        """Reset tracking state"""
        self.accumulated_q = 0.0
        self.accumulated_p = 0.0
        self.logical_x_errors = 0
        self.logical_z_errors = 0
        self.total_rounds = 0
        self.error_history: List[LogicalErrorEvent] = []

    def update(self,
               syndrome_q: float,
               syndrome_p: float,
               correction_q: float,
               correction_p: float) -> Tuple[bool, bool]:
        """
        Update accumulated displacement and check for logical errors

        In each QEC round:
        1. We measure syndrome (noisy estimate of error mod lattice)
        2. We apply correction based on syndrome
        3. Residual = true_error - correction accumulates over rounds

        If we make a wrong decision (error was actually closer to a
        different lattice point), we accumulate ±λ/2 extra displacement.

        Args:
            syndrome_q: Measured q syndrome
            syndrome_p: Measured p syndrome
            correction_q: Applied q correction
            correction_p: Applied p correction

        Returns:
            (x_error, z_error): Whether each type of logical error occurred
        """
        self.total_rounds += 1

        # The difference between syndrome and correction is the residual
        # This accumulates if there are systematic biases or wrong decisions
        residual_q = syndrome_q - correction_q
        residual_p = syndrome_p - correction_p

        self.accumulated_q += residual_q
        self.accumulated_p += residual_p

        x_error = False
        z_error = False

        # Check for logical X error (accumulated q displacement too large)
        if abs(self.accumulated_q) > self.lattice / 2:
            self.logical_x_errors += 1
            x_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='X',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # Wrap accumulated value
            self.accumulated_q = np.mod(self.accumulated_q + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        # Check for logical Z error (accumulated p displacement too large)
        if abs(self.accumulated_p) > self.lattice / 2:
            self.logical_z_errors += 1
            z_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='Z',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # Wrap accumulated value
            self.accumulated_p = np.mod(self.accumulated_p + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        return x_error, z_error

    def update_from_qec_result(self, qec_result: Dict) -> Tuple[bool, bool]:
        """
        Update from QEC round result dictionary

        Args:
            qec_result: Output from GKPErrorCorrector.run_qec_round()

        Returns:
            (x_error, z_error)
        """
        syndrome = qec_result['syndrome']
        correction = qec_result['correction']
        return self.update(syndrome[0], syndrome[1], correction[0], correction[1])

    def get_total_logical_errors(self) -> int:
        """Get total number of logical errors"""
        return self.logical_x_errors + self.logical_z_errors

    def get_logical_error_rate(self) -> float:
        """Get logical error rate per round"""
        if self.total_rounds == 0:
            return 0.0
        return self.get_total_logical_errors() / self.total_rounds

    def get_x_error_rate(self) -> float:
        """Get X logical error rate"""
        if self.total_rounds == 0:
            return 0.0
        return self.logical_x_errors / self.total_rounds

    def get_z_error_rate(self) -> float:
        """Get Z logical error rate"""
        if self.total_rounds == 0:
            return 0.0
        return self.logical_z_errors / self.total_rounds

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'total_rounds': self.total_rounds,
            'logical_x_errors': self.logical_x_errors,
            'logical_z_errors': self.logical_z_errors,
            'total_logical_errors': self.get_total_logical_errors(),
            'x_error_rate': self.get_x_error_rate(),
            'z_error_rate': self.get_z_error_rate(),
            'total_error_rate': self.get_logical_error_rate(),
            'accumulated_q': self.accumulated_q,
            'accumulated_p': self.accumulated_p,
        }

    def get_error_times(self) -> List[int]:
        """Get list of timesteps when logical errors occurred"""
        return [event.timestep for event in self.error_history]


class WindowedErrorTracker:
    """
    Tracks logical error rate over sliding windows

    Useful for detecting when performance degrades due to drift.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of sliding window for rate calculation
        """
        self.window_size = window_size
        self.tracker = LogicalErrorTracker()
        self.window_errors: List[int] = []  # Errors in each round (0 or 1)

    def reset(self):
        """Reset all tracking"""
        self.tracker.reset()
        self.window_errors = []

    def update(self,
               syndrome_q: float,
               syndrome_p: float,
               correction_q: float,
               correction_p: float) -> float:
        """
        Update and return windowed error rate

        Returns:
            Current windowed error rate
        """
        x_err, z_err = self.tracker.update(
            syndrome_q, syndrome_p, correction_q, correction_p
        )

        # Record if error occurred this round
        error_occurred = 1 if (x_err or z_err) else 0
        self.window_errors.append(error_occurred)

        # Keep only last window_size rounds
        if len(self.window_errors) > self.window_size:
            self.window_errors.pop(0)

        return self.get_windowed_error_rate()

    def get_windowed_error_rate(self) -> float:
        """Get error rate over current window"""
        if len(self.window_errors) == 0:
            return 0.0
        return sum(self.window_errors) / len(self.window_errors)

    def is_performance_degraded(self, threshold: float = 0.1) -> bool:
        """
        Check if error rate exceeds threshold

        Args:
            threshold: Error rate threshold

        Returns:
            True if current windowed error rate exceeds threshold
        """
        return self.get_windowed_error_rate() > threshold


class ExperimentErrorTracker:
    """
    Tracks errors across an entire experiment with multiple configurations

    Records error rates for different decoder configurations and
    noise conditions.
    """

    def __init__(self):
        self.results: List[Dict] = []
        self.current_tracker: Optional[LogicalErrorTracker] = None

    def start_configuration(self, config_name: str, params: Dict):
        """Start tracking a new configuration"""
        self.current_tracker = LogicalErrorTracker()
        self.current_config = {
            'name': config_name,
            'params': params,
        }

    def update(self, syndrome: np.ndarray, correction: np.ndarray) -> Tuple[bool, bool]:
        """Update current configuration's tracker"""
        if self.current_tracker is None:
            raise RuntimeError("No configuration started")
        return self.current_tracker.update(
            syndrome[0], syndrome[1], correction[0], correction[1]
        )

    def end_configuration(self):
        """End current configuration and save results"""
        if self.current_tracker is not None:
            result = {
                **self.current_config,
                'statistics': self.current_tracker.get_statistics(),
            }
            self.results.append(result)
            self.current_tracker = None

    def get_all_results(self) -> List[Dict]:
        """Get results from all configurations"""
        return self.results

    def get_summary(self) -> Dict:
        """Get summary statistics across all configurations"""
        if not self.results:
            return {}

        error_rates = [r['statistics']['total_error_rate'] for r in self.results]

        return {
            'n_configurations': len(self.results),
            'mean_error_rate': np.mean(error_rates),
            'std_error_rate': np.std(error_rates),
            'min_error_rate': np.min(error_rates),
            'max_error_rate': np.max(error_rates),
            'best_config': self.results[np.argmin(error_rates)]['name'],
            'worst_config': self.results[np.argmax(error_rates)]['name'],
        }


def simulate_error_accumulation(n_rounds: int,
                                 sigma_error: float,
                                 sigma_measurement: float,
                                 gain: float = 1.0) -> Dict:
    """
    Simulate error accumulation over many rounds

    This is a simplified simulation for analysis purposes.

    Args:
        n_rounds: Number of QEC rounds
        sigma_error: Standard deviation of errors per round
        sigma_measurement: Standard deviation of measurement noise
        gain: Decoder gain factor

    Returns:
        Simulation statistics
    """
    tracker = LogicalErrorTracker()

    for _ in range(n_rounds):
        # Random error this round
        error_q = np.random.normal(0, sigma_error)
        error_p = np.random.normal(0, sigma_error)

        # Noisy syndrome measurement
        syndrome_q = (np.mod(error_q + LATTICE_CONST/2, LATTICE_CONST) - LATTICE_CONST/2
                      + np.random.normal(0, sigma_measurement))
        syndrome_p = (np.mod(error_p + LATTICE_CONST/2, LATTICE_CONST) - LATTICE_CONST/2
                      + np.random.normal(0, sigma_measurement))

        # Correction
        correction_q = gain * syndrome_q
        correction_p = gain * syndrome_p

        # Note: This simplified model doesn't perfectly match full QEC
        # but captures the error accumulation dynamics
        tracker.update(syndrome_q, syndrome_p, correction_q, correction_p)

    return tracker.get_statistics()
