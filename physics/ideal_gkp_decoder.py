"""理想 square-GKP syndrome-level 解码器。

本模块处理 square-GKP 的理想 1D/2D syndrome-level 位移模型，并冻结仓库当前使用的
无量纲约定：

``lambda = LATTICE_CONST = sqrt(2*pi)``

``lambda`` 同时作为 standard-binning correction cell 的间距和 centered modular
syndrome 的周期。相邻 cell 的逻辑陪集奇偶性相反；因此把位移 ``x`` 分到最近的
``k * lambda`` 后，``k mod 2`` 就是该轴的逻辑翻转标签。不同文献常使用
``sqrt(pi)``/``2*sqrt(pi)`` 的 canonical-quadrature 写法；比较数值前必须先做
quadrature scaling，不能直接混用常数。

二维路径包含任意严格正定 Gaussian covariance 下的四 logical cosets；它仍不包含
finite-energy envelope、measurement noise、outer code 或多轮控制，这些能力由后续
task 单独实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, erfc, exp, isfinite, log, pi, sqrt
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LATTICE_CONST


ScalarOrArray = Union[float, NDArray[np.float64]]
IntOrArray = Union[int, NDArray[np.int64]]
BoolOrArray = Union[bool, NDArray[np.bool_]]
_MAX_ALIAS_EVALUATIONS = 10_000_000


@dataclass(frozen=True)
class StandardBinningResult:
    """单轴 standard-binning 判决结果。

    ``correction`` 是需要从真实位移中减去的 centered syndrome；满足
    ``displacement = correction + lattice_index * lattice``（浮点误差内）。
    ``logical_parity`` 为 ``lattice_index mod 2``，1 表示该轴发生逻辑翻转。
    """

    syndrome: ScalarOrArray
    correction: ScalarOrArray
    lattice_index: IntOrArray
    logical_parity: IntOrArray
    logical_flip: BoolOrArray


@dataclass(frozen=True)
class MAPDecodeResult:
    """单轴 periodic-Gaussian MAP hard/soft 判决。

    LLR 统一定义为 ``log P(even|s) - log P(odd|s)``；因此 LLR 非负时
    ``parity=0``，负时 ``parity=1``，精确 tie 固定选择 even。``confidence``
    是 ``|P(even|s)-P(odd|s)|``，范围为 [0, 1]。
    """

    syndrome: ScalarOrArray
    parity: IntOrArray
    logical_flip: BoolOrArray
    llr: ScalarOrArray
    posterior_even: ScalarOrArray
    posterior_odd: ScalarOrArray
    confidence: ScalarOrArray
    log_likelihood_even: ScalarOrArray
    log_likelihood_odd: ScalarOrArray


@dataclass(frozen=True)
class MAPDecode2DResult:
    """二维四 logical-coset MAP 判决及完整 posterior。

    ``parity[..., 0]``/``parity[..., 1]`` 分别是 q/p parity；
    ``logical_class = 2 * parity_q + parity_p``，tie 按 ``00, 01, 10, 11``
    顺序选择。``posterior[..., bq, bp]`` 与 ``log_likelihoods`` 使用相同索引。
    """

    syndrome: NDArray[np.float64]
    parity: NDArray[np.int64]
    logical_flips: NDArray[np.bool_]
    logical_class: IntOrArray
    posterior: NDArray[np.float64]
    confidence: ScalarOrArray
    log_likelihoods: NDArray[np.float64]
    method: Literal["joint", "independent"]


def _validate_lattice(lattice: float) -> float:
    value = float(lattice)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("lattice must be finite and positive")
    return value


def _scalarize(value: np.ndarray, scalar_input: bool):
    if not scalar_input:
        return value
    item = value.item()
    if np.issubdtype(value.dtype, np.bool_):
        return bool(item)
    if np.issubdtype(value.dtype, np.integer):
        return int(item)
    return float(item)


def _nearest_lattice_index(coordinate: np.ndarray, *, field: str) -> np.ndarray:
    """安全计算 nearest integer lattice index，拒绝 int64 静默溢出。"""

    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{field} is too large relative to lattice")
    # 留出两位余量，供 boundary 修正和 alias offsets 使用。超过该范围的 index
    # 既无法可靠编码为 int64，也没有可重复的软件/RTL 表示，因此显式拒绝。
    safe_limit = float(2**62)
    if np.any(coordinate <= -safe_limit) or np.any(coordinate >= safe_limit):
        raise ValueError(f"{field} requires a lattice index outside the supported int64 range")
    return np.floor(coordinate + 0.5).astype(np.int64)


def _validate_alias_workload(batch_size: int, aliases_per_item: int) -> None:
    """在分配 alias 张量前阻止无界内存消耗；大 batch 应由调用方分块。"""

    if aliases_per_item <= 0 or batch_size < 0:
        raise ValueError("invalid alias workload")
    if aliases_per_item > _MAX_ALIAS_EVALUATIONS or (
        batch_size > 0 and aliases_per_item > _MAX_ALIAS_EVALUATIONS // batch_size
    ):
        raise ValueError(
            "alias workload exceeds the safety limit; reduce noise scale or decode the batch in chunks"
        )


def standard_binning_1d(
    displacement: ArrayLike,
    *,
    lattice: float = LATTICE_CONST,
) -> StandardBinningResult:
    """对一个或一批单轴位移执行 nearest-cell standard binning。

    判决区间采用半开约定
    ``[(k-1/2)lambda, (k+1/2)lambda)``。因此正边界 ``+lambda/2``
    分到 ``k=1``，负边界 ``-lambda/2`` 分到 ``k=0``。连续 Gaussian 噪声
    在精确边界上的概率为零；显式约定用于保证软件/RTL 可重复。

    Parameters
    ----------
    displacement:
        标量或任意形状的有限实数数组。
    lattice:
        correction-cell spacing / syndrome period，默认 ``sqrt(2*pi)``。
    """

    spacing = _validate_lattice(lattice)
    values = np.asarray(displacement, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("displacement must contain only finite values")

    scalar_input = values.ndim == 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinate = values / spacing
    lattice_index = _nearest_lattice_index(coordinate, field="displacement")
    syndrome = values - lattice_index.astype(np.float64) * spacing

    # 消除边界附近浮点舍入导致的 +lambda/2，保持 [-lambda/2, lambda/2)。
    upper = spacing / 2.0
    boundary_mask = syndrome >= upper
    if np.any(boundary_mask):
        syndrome = np.where(boundary_mask, syndrome - spacing, syndrome)
        lattice_index = np.where(boundary_mask, lattice_index + 1, lattice_index)

    logical_parity = np.mod(lattice_index, 2).astype(np.int64)
    logical_flip = logical_parity.astype(bool)

    return StandardBinningResult(
        syndrome=_scalarize(syndrome, scalar_input),
        correction=_scalarize(syndrome.copy(), scalar_input),
        lattice_index=_scalarize(lattice_index, scalar_input),
        logical_parity=_scalarize(logical_parity, scalar_input),
        logical_flip=_scalarize(logical_flip, scalar_input),
    )


def _validate_centered_syndrome(syndrome: ArrayLike, lattice: float) -> np.ndarray:
    values = np.asarray(syndrome, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("syndrome must contain only finite values")
    half = lattice / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError("syndrome must lie in the half-open interval [-lattice/2, lattice/2)")
    return values


def _validate_gaussian_model(
    sigma: float,
    mean: ArrayLike,
    syndrome: np.ndarray,
) -> tuple[float, np.ndarray]:
    noise_sigma = float(sigma)
    if not isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("sigma must be finite and positive for MAP likelihoods")
    mean_values = np.asarray(mean, dtype=np.float64)
    if not np.all(np.isfinite(mean_values)):
        raise ValueError("mean must contain only finite values")
    try:
        broadcast_mean = np.broadcast_to(mean_values, syndrome.shape)
    except ValueError as exc:
        raise ValueError("mean must be broadcast-compatible with syndrome") from exc
    return noise_sigma, broadcast_mean


def _logsumexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _coset_log_likelihoods_1d(
    syndrome: ArrayLike,
    sigma: float,
    *,
    mean: ArrayLike,
    lattice: float,
    tail_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = _validate_lattice(lattice)
    values = _validate_centered_syndrome(syndrome, spacing)
    noise_sigma, mean_values = _validate_gaussian_model(sigma, mean, values)
    if not isfinite(tail_sigma) or tail_sigma <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")

    # 以每个 observation 最接近 mean 的 alias 为中心截断；这样 mean 可跨多个 cell，
    # 而 odd/even parity 仍由全局 lattice index 决定。
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinate = (mean_values - values) / spacing
    nearest = _nearest_lattice_index(coordinate, field="mean-syndrome offset")
    radius = max(2, int(ceil(tail_sigma * noise_sigma / spacing)) + 2)
    _validate_alias_workload(values.size, 2 * radius + 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    aliases = nearest[..., np.newaxis] + offsets
    residual = values[..., np.newaxis] + aliases * spacing - mean_values[..., np.newaxis]
    log_weights = (
        -0.5 * (residual / noise_sigma) ** 2
        - log(noise_sigma * sqrt(2.0 * pi))
    )

    even_terms = np.where(np.mod(aliases, 2) == 0, log_weights, -np.inf)
    odd_terms = np.where(np.mod(aliases, 2) == 1, log_weights, -np.inf)
    log_even = _logsumexp(even_terms, axis=-1)
    log_odd = _logsumexp(odd_terms, axis=-1)
    return values, log_even, log_odd


def coset_likelihood_1d(
    syndrome: ArrayLike,
    sigma: float,
    parity: Literal[0, 1],
    *,
    mean: ArrayLike = 0.0,
    lattice: float = LATTICE_CONST,
    tail_sigma: float = 10.0,
    log_output: bool = False,
) -> ScalarOrArray:
    r"""计算 centered syndrome 对偶/奇 logical coset 的 periodic Gaussian 似然。

    .. math::

       L_b(s)=\sum_{k\bmod 2=b}
       \mathcal N(s+k\lambda;\mu,\sigma^2),\qquad b\in\{0,1\}.

    ``log_output=True`` 返回稳定的 log-likelihood，适用于 rare-event/小噪声。
    输入必须已经位于 ``[-lambda/2, lambda/2)``，函数不会静默 wrap 并改变
    logical parity 语义。
    """

    if parity not in (0, 1):
        raise ValueError("parity must be 0 (even) or 1 (odd)")
    values, log_even, log_odd = _coset_log_likelihoods_1d(
        syndrome,
        sigma,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    selected = log_even if parity == 0 else log_odd
    output = selected if log_output else np.exp(selected)
    return _scalarize(np.asarray(output), values.ndim == 0)


def llr_1d(
    syndrome: ArrayLike,
    sigma: float,
    *,
    mean: ArrayLike = 0.0,
    lattice: float = LATTICE_CONST,
    prior_even: float = 0.5,
    tail_sigma: float = 10.0,
) -> ScalarOrArray:
    """返回 ``log P(even|s) - log P(odd|s)``。

    ``prior_even`` 允许显式 logical prior；默认等先验。正 LLR 支持 even，
    负 LLR 支持 odd。函数内部始终使用 log-sum-exp，不由 raw likelihood 相除。
    """

    prior = float(prior_even)
    if not isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("prior_even must lie strictly between 0 and 1")
    values, log_even, log_odd = _coset_log_likelihoods_1d(
        syndrome,
        sigma,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    llr = log_even - log_odd + log(prior) - log(1.0 - prior)
    return _scalarize(np.asarray(llr), values.ndim == 0)


def _posterior_even_from_llr(llr: np.ndarray) -> np.ndarray:
    positive = llr >= 0.0
    output = np.empty_like(llr, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-llr[positive]))
    exp_value = np.exp(llr[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def map_decode_1d(
    syndrome: ArrayLike,
    sigma: float,
    *,
    mean: ArrayLike = 0.0,
    lattice: float = LATTICE_CONST,
    prior_even: float = 0.5,
    tail_sigma: float = 10.0,
) -> MAPDecodeResult:
    """执行 periodic-Gaussian MAP hard decision，并同时返回 soft posterior。"""

    spacing = _validate_lattice(lattice)
    values = _validate_centered_syndrome(syndrome, spacing)
    _, log_even, log_odd = _coset_log_likelihoods_1d(
        values,
        sigma,
        mean=mean,
        lattice=spacing,
        tail_sigma=tail_sigma,
    )
    prior = float(prior_even)
    if not isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("prior_even must lie strictly between 0 and 1")
    llr = log_even - log_odd + log(prior) - log(1.0 - prior)
    llr_array = np.asarray(llr, dtype=np.float64)
    posterior_even = _posterior_even_from_llr(llr_array)
    posterior_odd = 1.0 - posterior_even
    parity = (llr_array < 0.0).astype(np.int64)  # exact tie -> even
    logical_flip = parity.astype(bool)
    confidence = np.abs(posterior_even - posterior_odd)
    scalar_input = values.ndim == 0

    return MAPDecodeResult(
        syndrome=_scalarize(values, scalar_input),
        parity=_scalarize(parity, scalar_input),
        logical_flip=_scalarize(logical_flip, scalar_input),
        llr=_scalarize(llr_array, scalar_input),
        posterior_even=_scalarize(posterior_even, scalar_input),
        posterior_odd=_scalarize(posterior_odd, scalar_input),
        confidence=_scalarize(confidence, scalar_input),
        log_likelihood_even=_scalarize(np.asarray(log_even), scalar_input),
        log_likelihood_odd=_scalarize(np.asarray(log_odd), scalar_input),
    )


def decode_1d(
    value: ArrayLike,
    *,
    mode: Literal["standard", "map", "soft"] = "standard",
    sigma: float | None = None,
    mean: ArrayLike = 0.0,
    lattice: float = LATTICE_CONST,
    prior_even: float = 0.5,
    tail_sigma: float = 10.0,
) -> StandardBinningResult | MAPDecodeResult | ScalarOrArray:
    """统一的 1D 三模式入口。

    - ``standard``：``value`` 是 raw displacement，返回 nearest-cell 结果；
    - ``map``：``value`` 是 centered syndrome，返回 hard+soft ``MAPDecodeResult``；
    - ``soft``：``value`` 是 centered syndrome，返回 even-vs-odd LLR。

    MAP/soft 模式必须显式提供正 ``sigma``，避免静默使用任意噪声假设。
    """

    if mode == "standard":
        if sigma is not None:
            raise ValueError("sigma is not used in standard mode; omit it explicitly")
        return standard_binning_1d(value, lattice=lattice)
    if mode == "map":
        if sigma is None:
            raise ValueError("sigma is required in map mode")
        return map_decode_1d(
            value,
            sigma,
            mean=mean,
            lattice=lattice,
            prior_even=prior_even,
            tail_sigma=tail_sigma,
        )
    if mode == "soft":
        if sigma is None:
            raise ValueError("sigma is required in soft mode")
        return llr_1d(
            value,
            sigma,
            mean=mean,
            lattice=lattice,
            prior_even=prior_even,
            tail_sigma=tail_sigma,
        )
    raise ValueError("mode must be 'standard', 'map', or 'soft'")


def covariance_from_sigmas(
    sigma_q: float,
    sigma_p: float,
    rho: float = 0.0,
) -> NDArray[np.float64]:
    r"""由 marginal sigmas 与 correlation coefficient 构造 2D covariance。

    .. math::

       \Sigma=\begin{pmatrix}
       \sigma_q^2 & \rho\sigma_q\sigma_p\\
       \rho\sigma_q\sigma_p & \sigma_p^2
       \end{pmatrix}.

    MAP density 要求严格正定，因此 ``|rho| < 1``；奇异的 ``rho=+/-1`` 不会被
    静默加 jitter。
    """

    q_sigma = float(sigma_q)
    p_sigma = float(sigma_p)
    correlation = float(rho)
    if not isfinite(q_sigma) or q_sigma <= 0.0:
        raise ValueError("sigma_q must be finite and positive")
    if not isfinite(p_sigma) or p_sigma <= 0.0:
        raise ValueError("sigma_p must be finite and positive")
    if not isfinite(correlation) or not (-1.0 < correlation < 1.0):
        raise ValueError("rho must be finite and lie strictly between -1 and 1")

    covariance = np.array(
        [
            [q_sigma * q_sigma, correlation * q_sigma * p_sigma],
            [correlation * q_sigma * p_sigma, p_sigma * p_sigma],
        ],
        dtype=np.float64,
    )
    _validate_covariance_2d(covariance)
    return covariance


def _validate_covariance_2d(
    covariance: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("covariance must have shape (2, 2)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain only finite values")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14 * scale):
        raise ValueError("covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(matrix)
        inverse = np.linalg.solve(matrix, np.eye(2, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be strictly positive definite") from exc
    if not np.all(np.isfinite(inverse)):
        raise ValueError("covariance is numerically singular")
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    sigma_q = sqrt(float(matrix[0, 0]))
    sigma_p = sqrt(float(matrix[1, 1]))
    return matrix, inverse, log_determinant, sigma_q, sigma_p


def _validate_syndrome_2d(
    syndrome: ArrayLike,
    mean: ArrayLike,
    lattice: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(syndrome, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("syndrome must have shape (2,) or (..., 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("syndrome must contain only finite values")
    half = lattice / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError(
            "each syndrome coordinate must lie in [-lattice/2, lattice/2)"
        )

    mean_values = np.asarray(mean, dtype=np.float64)
    if not np.all(np.isfinite(mean_values)):
        raise ValueError("mean must contain only finite values")
    try:
        broadcast_mean = np.broadcast_to(mean_values, values.shape)
    except ValueError as exc:
        raise ValueError("mean must be broadcast-compatible with syndrome") from exc
    return values, broadcast_mean


def _coset_log_likelihoods_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    *,
    mean: ArrayLike,
    lattice: float,
    tail_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = _validate_lattice(lattice)
    values, mean_values = _validate_syndrome_2d(syndrome, mean, spacing)
    _, inverse, log_determinant, sigma_q, sigma_p = _validate_covariance_2d(covariance)
    if not isfinite(tail_sigma) or tail_sigma <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinates = (mean_values - values) / spacing
    nearest = _nearest_lattice_index(coordinates, field="mean-syndrome offset")

    # 以 marginal Gaussian tails 构造 rectangle。即便 q/p 相关，遗漏概率也由
    # 两个 marginal tail 的 union bound 控制，不依赖 rho 的符号。
    radius_q = max(2, int(ceil(tail_sigma * sigma_q / spacing)) + 2)
    radius_p = max(2, int(ceil(tail_sigma * sigma_p / spacing)) + 2)
    aliases_per_item = (2 * radius_q + 1) * (2 * radius_p + 1)
    batch_size = int(np.prod(values.shape[:-1], dtype=np.int64))
    _validate_alias_workload(batch_size, aliases_per_item)
    offsets_q = np.arange(-radius_q, radius_q + 1, dtype=np.int64)
    offsets_p = np.arange(-radius_p, radius_p + 1, dtype=np.int64)
    aliases_q = nearest[..., 0][..., np.newaxis, np.newaxis] + offsets_q[:, np.newaxis]
    aliases_p = nearest[..., 1][..., np.newaxis, np.newaxis] + offsets_p[np.newaxis, :]
    residual_q = (
        values[..., 0][..., np.newaxis, np.newaxis]
        + aliases_q * spacing
        - mean_values[..., 0][..., np.newaxis, np.newaxis]
    )
    residual_p = (
        values[..., 1][..., np.newaxis, np.newaxis]
        + aliases_p * spacing
        - mean_values[..., 1][..., np.newaxis, np.newaxis]
    )
    quadratic = (
        inverse[0, 0] * residual_q * residual_q
        + 2.0 * inverse[0, 1] * residual_q * residual_p
        + inverse[1, 1] * residual_p * residual_p
    )
    log_weights = -0.5 * quadratic - log(2.0 * pi) - 0.5 * log_determinant

    rows = []
    for parity_q in (0, 1):
        columns = []
        for parity_p in (0, 1):
            mask = (
                (np.mod(aliases_q, 2) == parity_q)
                & (np.mod(aliases_p, 2) == parity_p)
            )
            terms = np.where(mask, log_weights, -np.inf)
            columns.append(_logsumexp(terms, axis=(-2, -1)))
        rows.append(np.stack(columns, axis=-1))
    log_likelihoods = np.stack(rows, axis=-2)
    return values, log_likelihoods


def coset_likelihood_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    parity: tuple[int, int],
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    tail_sigma: float = 10.0,
    log_output: bool = False,
) -> ScalarOrArray:
    r"""计算二维 correlated Gaussian 的指定 logical-coset likelihood。

    .. math::

       L_{ab}(s_q,s_p)=\sum_{k_q\bmod2=a}\sum_{k_p\bmod2=b}
       \mathcal N_2(s+\lambda k;\mu,\Sigma).

    ``parity=(b_q,b_p)``；输入可为单个 ``(2,)`` 或 batch ``(...,2)``。
    """

    try:
        logical_parity = tuple(parity)
    except TypeError as exc:
        raise ValueError("parity must be a length-2 pair containing only 0 or 1") from exc
    if len(logical_parity) != 2 or any(value not in (0, 1) for value in logical_parity):
        raise ValueError("parity must be a length-2 pair containing only 0 or 1")
    parity_q, parity_p = (int(logical_parity[0]), int(logical_parity[1]))
    values, log_likelihoods = _coset_log_likelihoods_2d(
        syndrome,
        covariance,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    selected = log_likelihoods[..., parity_q, parity_p]
    output = selected if log_output else np.exp(selected)
    return _scalarize(np.asarray(output), values.ndim == 1)


def _validate_joint_prior(prior: ArrayLike | None) -> np.ndarray:
    if prior is None:
        return np.full((2, 2), 0.25, dtype=np.float64)
    probabilities = np.asarray(prior, dtype=np.float64)
    if probabilities.shape != (2, 2):
        raise ValueError("prior must have shape (2, 2)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("all joint prior entries must be finite and strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(probabilities))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("joint prior sum must be finite and positive")
    return probabilities / total


def _build_map_decode_2d_result(
    values: np.ndarray,
    log_likelihoods: np.ndarray,
    prior: np.ndarray,
    *,
    method: Literal["joint", "independent"],
) -> MAPDecode2DResult:
    log_scores = log_likelihoods + np.log(prior)
    log_evidence = _logsumexp(log_scores, axis=(-2, -1))
    posterior = np.exp(log_scores - log_evidence[..., np.newaxis, np.newaxis])
    flat_posterior = posterior.reshape(posterior.shape[:-2] + (4,))
    logical_class = np.argmax(flat_posterior, axis=-1).astype(np.int64)
    parity = np.stack((logical_class // 2, logical_class % 2), axis=-1).astype(np.int64)
    ordered = np.sort(flat_posterior, axis=-1)
    confidence = ordered[..., -1] - ordered[..., -2]
    scalar_batch = values.ndim == 1

    return MAPDecode2DResult(
        syndrome=values.copy(),
        parity=parity,
        logical_flips=parity.astype(bool),
        logical_class=_scalarize(np.asarray(logical_class), scalar_batch),
        posterior=posterior,
        confidence=_scalarize(np.asarray(confidence), scalar_batch),
        log_likelihoods=log_likelihoods,
        method=method,
    )


def map_decode_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    prior: ArrayLike | None = None,
    tail_sigma: float = 10.0,
) -> MAPDecode2DResult:
    """执行 correlated-Gaussian joint MAP，返回四类 posterior 与 hard decision。"""

    validated_prior = _validate_joint_prior(prior)
    values, log_likelihoods = _coset_log_likelihoods_2d(
        syndrome,
        covariance,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    return _build_map_decode_2d_result(
        values,
        log_likelihoods,
        validated_prior,
        method="joint",
    )


def independent_map_decode_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    prior_even_q: float = 0.5,
    prior_even_p: float = 0.5,
    tail_sigma: float = 10.0,
) -> MAPDecode2DResult:
    """忽略 off-diagonal covariance，以两个 marginal 1D MAP 形成对照解码器。"""

    spacing = _validate_lattice(lattice)
    values, mean_values = _validate_syndrome_2d(syndrome, mean, spacing)
    _, _, _, sigma_q, sigma_p = _validate_covariance_2d(covariance)
    prior_q = float(prior_even_q)
    prior_p = float(prior_even_p)
    if not isfinite(prior_q) or not (0.0 < prior_q < 1.0):
        raise ValueError("prior_even_q must lie strictly between 0 and 1")
    if not isfinite(prior_p) or not (0.0 < prior_p < 1.0):
        raise ValueError("prior_even_p must lie strictly between 0 and 1")

    _, log_q_even, log_q_odd = _coset_log_likelihoods_1d(
        values[..., 0],
        sigma_q,
        mean=mean_values[..., 0],
        lattice=spacing,
        tail_sigma=tail_sigma,
    )
    _, log_p_even, log_p_odd = _coset_log_likelihoods_1d(
        values[..., 1],
        sigma_p,
        mean=mean_values[..., 1],
        lattice=spacing,
        tail_sigma=tail_sigma,
    )
    q_logs = np.stack((log_q_even, log_q_odd), axis=-1)
    p_logs = np.stack((log_p_even, log_p_odd), axis=-1)
    log_likelihoods = q_logs[..., :, np.newaxis] + p_logs[..., np.newaxis, :]
    prior = np.array(
        [
            [prior_q * prior_p, prior_q * (1.0 - prior_p)],
            [(1.0 - prior_q) * prior_p, (1.0 - prior_q) * (1.0 - prior_p)],
        ],
        dtype=np.float64,
    )
    return _build_map_decode_2d_result(
        values,
        log_likelihoods,
        prior,
        method="independent",
    )


def _flip_probability_interval_sum(
    sigma: float,
    lattice: float,
    *,
    tail_sigma: float,
) -> float:
    """以 odd-cell Gaussian interval sum 计算逻辑翻转概率。"""

    # 正半轴 odd cells: [(2m+1/2)lambda, (2m+3/2)lambda], m>=0；
    # 乘二后的 Gaussian interval 正好化为两个 erfc 之差。
    last_m = max(0, ceil((tail_sigma * sigma / lattice - 1.5) / 2.0))
    probability = 0.0
    scale = sigma * sqrt(2.0)
    for m in range(last_m + 1):
        lower = (2.0 * m + 0.5) * lattice
        upper = (2.0 * m + 1.5) * lattice
        probability += erfc(lower / scale) - erfc(upper / scale)
    return probability


def _flip_probability_fourier(
    sigma: float,
    lattice: float,
    *,
    tolerance: float,
    max_terms: int,
) -> float:
    """以 ``sign(cos(pi*x/lambda))`` 的 Fourier 展开计算翻转概率。"""

    balance = 0.0  # P(even cell) - P(odd cell)
    ratio = pi * sigma / lattice
    for m in range(max_terms):
        harmonic = 2 * m + 1
        term = ((-1.0) ** m) * exp(-0.5 * (harmonic * ratio) ** 2) / harmonic
        balance += term
        if abs(term) < tolerance:
            break
    else:
        raise RuntimeError("Fourier series did not converge within max_terms")
    return 0.5 * (1.0 - (4.0 / pi) * balance)


def gaussian_logical_flip_probability(
    sigma: float,
    *,
    lattice: float = LATTICE_CONST,
    method: Literal["auto", "interval", "fourier"] = "auto",
    tail_sigma: float = 10.0,
    tolerance: float = 1.0e-15,
    max_terms: int = 100_000,
) -> float:
    r"""返回零均值 Gaussian displacement 下的单轴逻辑翻转概率。

    对 ``X ~ Normal(0, sigma^2)``，standard binning 的错误区间是所有奇数
    correction cells：

    .. math::

       P_L = \sum_{k\in 2\mathbb{Z}+1}
             \int_{(k-1/2)\lambda}^{(k+1/2)\lambda}
             \frac{e^{-x^2/(2\sigma^2)}}{\sqrt{2\pi}\sigma}\,dx.

    ``interval`` 使用 erfc interval sum，对小 ``sigma/lambda`` 的稀有事件稳定；
    ``fourier`` 使用 square-wave Fourier series，对宽分布快速收敛；``auto``
    在二者间选择。截断误差受 ``P(|X| > tail_sigma*sigma)`` 控制。
    """

    spacing = _validate_lattice(lattice)
    noise_sigma = float(sigma)
    if not isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if noise_sigma == 0.0:
        return 0.0
    if not isfinite(tail_sigma) or tail_sigma <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_terms, int) or max_terms <= 0:
        raise ValueError("max_terms must be a positive integer")
    if method not in {"auto", "interval", "fourier"}:
        raise ValueError("method must be 'auto', 'interval', or 'fourier'")

    selected = method
    if selected == "auto":
        selected = "interval" if noise_sigma / spacing < 0.25 else "fourier"

    if selected == "interval":
        probability = _flip_probability_interval_sum(
            noise_sigma,
            spacing,
            tail_sigma=tail_sigma,
        )
    else:
        probability = _flip_probability_fourier(
            noise_sigma,
            spacing,
            tolerance=tolerance,
            max_terms=max_terms,
        )

    # 截断和浮点误差只允许产生极小越界；物理结果严格落在 [0, 1/2]。
    return float(np.clip(probability, 0.0, 0.5))


__all__ = [
    "StandardBinningResult",
    "MAPDecodeResult",
    "MAPDecode2DResult",
    "standard_binning_1d",
    "gaussian_logical_flip_probability",
    "coset_likelihood_1d",
    "llr_1d",
    "map_decode_1d",
    "decode_1d",
    "covariance_from_sigmas",
    "coset_likelihood_2d",
    "map_decode_2d",
    "independent_map_decode_2d",
]
