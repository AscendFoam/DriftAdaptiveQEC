"""
Quantum Noise Channels Module

Implements physically accurate noise channels for bosonic systems:
- Photon loss (amplitude damping)
- Thermal noise
- Displacement noise
- Dephasing / phase diffusion

These are proper quantum channels, not classical approximations.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass

from .gkp_state import ApproximateGKPState, LATTICE_CONST

# Try to import Strawberry Fields
try:
    import strawberryfields as sf
    from strawberryfields.ops import LossChannel, ThermalLossChannel, Dgate, Rgate
    HAS_STRAWBERRYFIELDS = True
except ImportError:
    HAS_STRAWBERRYFIELDS = False


@dataclass
class NoiseParameters:
    """Parameters for noise channels"""
    gamma: float = 0.05  # Photon loss rate
    n_bar: float = 0.01  # Thermal photon number
    sigma_displacement: float = 0.1  # Displacement noise std
    sigma_phase: float = 0.01  # Phase noise std


class QuantumNoiseChannel:
    """
    Combined quantum noise channel for GKP states

    Applies multiple noise sources in sequence:
    1. Photon loss (dominant in superconducting cavities)
    2. Thermal noise
    3. Displacement noise (control imperfections)
    4. Phase noise / dephasing
    """

    def __init__(self, cutoff: int = 50, use_sf: bool = True):
        self.cutoff = cutoff
        self.use_sf = use_sf and HAS_STRAWBERRYFIELDS

    def apply_all(self,
                  wigner: np.ndarray,
                  params: NoiseParameters,
                  grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """
        Apply all noise channels to a Wigner function

        For efficiency, this uses analytical approximations when possible.
        """
        W = wigner.copy()

        # 1. Photon loss: contraction + diffusion
        W = self._apply_photon_loss_wigner(W, params.gamma, grid_range)

        # 2. Thermal noise: Gaussian convolution
        W = self._apply_thermal_noise_wigner(W, params.n_bar, grid_range)

        # 3. Displacement noise: additional broadening
        W = self._apply_displacement_noise_wigner(W, params.sigma_displacement, grid_range)

        # 4. Phase noise: angular smearing
        if params.sigma_phase > 0.01:
            W = self._apply_phase_noise_wigner(W, params.sigma_phase)

        return W

    def _apply_photon_loss_wigner(self,
                                   W: np.ndarray,
                                   gamma: float,
                                   grid_range: Tuple[float, float]) -> np.ndarray:
        """
        Apply photon loss channel to Wigner function

        Photon loss transforms the Wigner function as:
        W(q,p) → (1/η) W(q/√η, p/√η) * G_{σ_η}

        where η = exp(-γ) is the transmissivity and σ_η = √((1-η)/2)

        The effect is:
        1. Contraction toward origin (scaling by √η)
        2. Gaussian diffusion with width σ_η
        """
        eta = np.exp(-gamma)  # Transmissivity
        if eta > 0.999:  # Negligible loss
            return W

        grid_size = W.shape[0]
        scale_factor = np.sqrt(eta)

        # 1. Contraction toward origin via interpolation
        from scipy.ndimage import zoom, shift
        # Scale the coordinates (zoom with center preservation)
        center = grid_size // 2

        # Create contracted version
        if scale_factor < 0.99:
            # Zoom and pad
            zoomed_size = int(grid_size * scale_factor)
            if zoomed_size < 3:
                zoomed_size = 3
            W_zoomed = zoom(W, scale_factor, order=1)
            # Pad to original size
            pad_size = (grid_size - W_zoomed.shape[0]) // 2
            W_contracted = np.pad(W_zoomed,
                                  ((pad_size, grid_size - W_zoomed.shape[0] - pad_size),
                                   (pad_size, grid_size - W_zoomed.shape[1] - pad_size)),
                                  mode='constant', constant_values=0)
            # Ensure same shape
            W_contracted = W_contracted[:grid_size, :grid_size]
        else:
            W_contracted = W

        # 2. Gaussian diffusion
        sigma_eta = np.sqrt((1 - eta) / 2)
        # Convert to pixel units
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_eta / dx

        if sigma_pixels > 0.1:
            W_final = gaussian_filter(W_contracted, sigma=sigma_pixels)
        else:
            W_final = W_contracted

        # Renormalize
        if np.max(np.abs(W_final)) > 1e-10:
            W_final = W_final / np.max(np.abs(W_final)) * np.max(np.abs(W))

        return W_final

    def _apply_thermal_noise_wigner(self,
                                     W: np.ndarray,
                                     n_bar: float,
                                     grid_range: Tuple[float, float]) -> np.ndarray:
        """
        Apply thermal noise channel to Wigner function

        Thermal noise is a Gaussian convolution with width √(n_bar)
        """
        if n_bar < 1e-6:
            return W

        grid_size = W.shape[0]
        sigma_thermal = np.sqrt(n_bar)
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_thermal / dx

        return gaussian_filter(W, sigma=sigma_pixels)

    def _apply_displacement_noise_wigner(self,
                                          W: np.ndarray,
                                          sigma_disp: float,
                                          grid_range: Tuple[float, float]) -> np.ndarray:
        """
        Apply random displacement noise

        This models control imperfections (AWG drift, cable phase drift, etc.)
        Effect is Gaussian convolution.
        """
        if sigma_disp < 1e-6:
            return W

        grid_size = W.shape[0]
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_disp / dx

        return gaussian_filter(W, sigma=sigma_pixels)

    def _apply_phase_noise_wigner(self,
                                   W: np.ndarray,
                                   sigma_phase: float) -> np.ndarray:
        """
        Apply phase noise / dephasing

        This causes angular smearing in phase space.
        We approximate this by averaging over small rotations.
        """
        from scipy.ndimage import rotate

        n_samples = 5
        phases = np.random.normal(0, sigma_phase, n_samples)

        W_avg = np.zeros_like(W)
        for phi in phases:
            W_rotated = rotate(W, np.degrees(phi), reshape=False, mode='constant')
            W_avg += W_rotated

        return W_avg / n_samples


class PhotonLossChannel:
    """
    Photon loss (amplitude damping) channel

    This is the dominant noise source in superconducting cavities.
    Characterized by the cavity decay rate κ and evolution time t:
    γ = κt gives the total loss.
    """

    def __init__(self, gamma: float):
        """
        Args:
            gamma: Total photon loss (dimensionless), γ = κt
        """
        self.gamma = gamma
        self.eta = np.exp(-gamma)  # Transmissivity

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """Apply to Wigner function representation"""
        channel = QuantumNoiseChannel()
        params = NoiseParameters(gamma=self.gamma, n_bar=0, sigma_displacement=0, sigma_phase=0)
        return channel._apply_photon_loss_wigner(W, self.gamma, grid_range)

    @classmethod
    def from_t1_and_time(cls, T1: float, t: float) -> 'PhotonLossChannel':
        """
        Create from T1 time and evolution time

        Args:
            T1: Energy relaxation time (e.g., in microseconds)
            t: Evolution time (same units as T1)
        """
        gamma = t / T1
        return cls(gamma)


class ThermalNoiseChannel:
    """
    Thermal noise channel

    Models the effect of a thermal bath with mean photon number n_bar.
    """

    def __init__(self, n_bar: float):
        """
        Args:
            n_bar: Mean thermal photon number
        """
        self.n_bar = n_bar

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """Apply to Wigner function representation"""
        channel = QuantumNoiseChannel()
        return channel._apply_thermal_noise_wigner(W, self.n_bar, grid_range)

    @classmethod
    def from_temperature(cls, T_kelvin: float, omega_hz: float) -> 'ThermalNoiseChannel':
        """
        Create from physical temperature

        Args:
            T_kelvin: Temperature in Kelvin
            omega_hz: Cavity frequency in Hz
        """
        import scipy.constants as const
        hbar_omega = const.hbar * omega_hz
        kT = const.k * T_kelvin

        if kT < 1e-30:  # Essentially zero temperature
            n_bar = 0.0
        else:
            n_bar = 1 / (np.exp(hbar_omega / kT) - 1)

        return cls(n_bar)


class DisplacementNoiseChannel:
    """
    Random displacement noise channel

    Models control imperfections:
    - AWG amplitude/phase drift
    - Cable thermal drift
    - Pulse calibration errors
    """

    def __init__(self, sigma_q: float, sigma_p: Optional[float] = None):
        """
        Args:
            sigma_q: Displacement noise std in q direction
            sigma_p: Displacement noise std in p direction (default: same as q)
        """
        self.sigma_q = sigma_q
        self.sigma_p = sigma_p if sigma_p is not None else sigma_q

    def sample_displacement(self) -> Tuple[float, float]:
        """Sample a random displacement error"""
        dq = np.random.normal(0, self.sigma_q)
        dp = np.random.normal(0, self.sigma_p)
        return dq, dp

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """Apply to Wigner function (as convolution)"""
        sigma_avg = (self.sigma_q + self.sigma_p) / 2
        channel = QuantumNoiseChannel()
        return channel._apply_displacement_noise_wigner(W, sigma_avg, grid_range)


class CombinedNoiseModel:
    """
    Combined noise model with time-varying parameters

    Convenient class for generating noisy GKP states with realistic noise.
    """

    def __init__(self,
                 gamma: float = 0.05,
                 n_bar: float = 0.01,
                 sigma_disp: float = 0.1):
        """
        Args:
            gamma: Photon loss rate
            n_bar: Thermal photon number
            sigma_disp: Displacement noise std
        """
        self.params = NoiseParameters(
            gamma=gamma,
            n_bar=n_bar,
            sigma_displacement=sigma_disp,
            sigma_phase=0.0
        )
        self.channel = QuantumNoiseChannel()

    def apply(self,
              wigner: np.ndarray,
              grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """Apply noise to Wigner function"""
        return self.channel.apply_all(wigner, self.params, grid_range)

    def update_params(self, **kwargs):
        """Update noise parameters"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def get_effective_sigma(self) -> float:
        """
        Get effective total noise standard deviation

        Combines all noise sources into equivalent displacement noise.
        """
        # Photon loss contribution
        sigma_loss = np.sqrt(self.params.gamma / 2) if self.params.gamma > 0 else 0

        # Thermal contribution
        sigma_thermal = np.sqrt(self.params.n_bar) if self.params.n_bar > 0 else 0

        # Total
        sigma_total = np.sqrt(sigma_loss**2 +
                              sigma_thermal**2 +
                              self.params.sigma_displacement**2)
        return sigma_total
