"""Audit manuscript/code symbol-boundary anchors for the submission draft.

This helper is intentionally narrow and read-only. It checks that selected
manuscript symbols and paper-facing metric terms are mechanically anchored in
the current source tree and source-data files. It does not run benchmarks,
import runtime modules, validate a physical noise model, or upgrade any
hardware/statistical evidence boundary.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs" / "figure_assets" / "submission_draft_python_figures"
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
TEX_PATH = ROOT / "docs" / "paper_notes" / "CNN_FPGA_GKP_submission_draft.tex"
REPORT_JSON = PAPER_MATERIALS / "submission_draft_symbol_boundary_audit.json"
REPORT_MD = PAPER_MATERIALS / "投稿稿符号边界机械审计报告.md"


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    topic: str
    paths: list[str]
    detail: str
    boundary: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def add_check(
    checks: list[Check],
    check_id: str,
    ok: bool,
    topic: str,
    paths: list[Path],
    detail: str,
    boundary: str,
    warn: bool = False,
) -> None:
    if warn:
        status = "WARN"
    else:
        status = "PASS" if ok else "FAIL"
    checks.append(
        Check(
            check_id=check_id,
            status=status,
            topic=topic,
            paths=[rel(path) for path in paths],
            detail=detail,
            boundary=boundary,
        )
    )


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def audit() -> dict[str, object]:
    checks: list[Check] = []

    tex = read_text(TEX_PATH)
    constants = read_text(ROOT / "physics" / "constants.py")
    syndrome = read_text(ROOT / "physics" / "syndrome_measurement.py")
    logical = read_text(ROOT / "physics" / "logical_tracking.py")
    fast_loop = read_text(ROOT / "cnn_fpga" / "runtime" / "fast_loop_emulator.py")
    param_mapper = read_text(ROOT / "cnn_fpga" / "decoder" / "param_mapper.py")
    hil_suite = read_text(ROOT / "cnn_fpga" / "benchmark" / "run_hil_suite.py")
    fig02 = read_csv(FIG_DIR / "source_data_fig02_main_results.csv")

    add_check(
        checks,
        "SB-A01",
        "\\lambda=\\sqrt{2\\pi}" in tex
        and "\\label{tab:physics-metric-boundary}" in tex
        and "\\finalLER{}" in tex
        and "logical-error proxy" in tex,
        "TeX symbol and metric boundary",
        [TEX_PATH],
        "Submission draft declares lambda=sqrt(2pi), includes the physics-metric boundary table, and names final_ler as a logical-error proxy.",
        "This anchors wording in the draft only; it does not prove physical correctness.",
    )

    add_check(
        checks,
        "SB-A09",
        contains_all(
            tex,
            [
                "\\textbf{Metric definition.}",
                "\\texttt{final\\_ler}=\\frac{N_X+N_Z}{T}",
                "final\\_ler\\_sd",
                "hardware logical-error measurement",
                "confidence",
                "significance test",
            ],
        ),
        "Metric definition box",
        [TEX_PATH],
        "Submission draft defines final_ler as (N_X+N_Z)/T, maps final_ler_mean/final_ler_sd to repeat-level source data, and states the non-hardware/non-inferential boundary.",
        "This is a manuscript definition check only; it does not validate the metric statistically or physically.",
    )

    add_check(
        checks,
        "SB-A02",
        bool(re.search(r"LATTICE_CONST\s*=\s*sqrt\(2\.0\s*\*\s*pi\)", constants)),
        "Shared GKP lattice constant",
        [ROOT / "physics" / "constants.py"],
        "physics/constants.py defines LATTICE_CONST as sqrt(2.0 * pi).",
        "This checks the shared constant value, not every downstream physical assumption.",
    )

    wrap_pattern = "np.mod(displacement + self.lattice / 2, self.lattice) - self.lattice / 2"
    realistic_wrap_pattern = "np.mod(true_displacement + self.lattice / 2"
    add_check(
        checks,
        "SB-A03",
        wrap_pattern in syndrome and realistic_wrap_pattern in syndrome and "self.lattice = LATTICE_CONST" in syndrome,
        "Syndrome wrap interval",
        [ROOT / "physics" / "syndrome_measurement.py"],
        "Ideal and realistic syndrome measurement paths wrap displacements into the shared half-lattice interval.",
        "This is a code-symbol consistency check; it is not a wrapped-posterior decoder analysis.",
    )

    add_check(
        checks,
        "SB-A04",
        contains_all(
            logical,
            [
                "from .constants import LATTICE_CONST",
                "self.lattice = LATTICE_CONST",
                "abs(self.accumulated_q) > self.lattice / 2",
                "abs(self.accumulated_p) > self.lattice / 2",
                "np.mod(self.accumulated_q + self.lattice / 2",
                "np.mod(self.accumulated_p + self.lattice / 2",
                "get_logical_error_rate",
            ],
        ),
        "Logical-error proxy rule",
        [ROOT / "physics" / "logical_tracking.py"],
        "LogicalErrorTracker uses half-lattice q/p thresholds, wraps accumulated residuals, and exposes get_logical_error_rate.",
        "This supports a software protocol metric only, not hardware logical-channel tomography.",
    )

    add_check(
        checks,
        "SB-A05",
        contains_all(
            fast_loop,
            [
                "from physics.constants import LATTICE_CONST",
                "from physics.logical_tracking import LogicalErrorTracker",
                "RealisticSyndromeMeasurement",
                "syndrome_limit: float = LATTICE_CONST / 2.0",
                "histogram_range_limit: float = LATTICE_CONST / 2.0",
                "self.tracker = LogicalErrorTracker()",
                '"final_logical_error_rate"',
            ],
        ),
        "Fast-loop metric and histogram boundary",
        [ROOT / "cnn_fpga" / "runtime" / "fast_loop_emulator.py"],
        "Fast-loop config uses half-lattice syndrome/histogram limits, realistic syndrome measurement, LogicalErrorTracker, and final_logical_error_rate output.",
        "This checks source anchors only; it does not run the emulator or validate fallback-free execution.",
    )

    add_check(
        checks,
        "SB-A06",
        contains_all(
            param_mapper,
            [
                "class NoisePrediction",
                "sigma: float",
                "mu_q: float",
                "mu_p: float",
                "theta_deg: float",
                "def map_prediction",
                "error_cov",
                "measurement_cov",
                "gain_eigvals_raw",
                "return DecoderRuntimeParams(K=k_next, b=b_next, metadata=metadata)",
            ],
        ),
        "Noise-state to affine-parameter mapping",
        [ROOT / "cnn_fpga" / "decoder" / "param_mapper.py"],
        "ParamMapper maps NoisePrediction(sigma, mu_q, mu_p, theta_deg) into K,b and records covariance/gain metadata.",
        "This supports the affine-map implementation story, not optimality or universal calibration claims.",
    )

    add_check(
        checks,
        "SB-A07",
        contains_all(
            hil_suite,
            [
                "def _build_mock_noise_provider",
                '"sigma": sigma_value',
                '"mu_q": float(mu_q_value)',
                '"mu_p": float(mu_p_value)',
                '"theta_deg": theta_value',
                '"metadata": {"signal_type": kind}',
                "backend_name == \"mock\"",
            ],
        ),
        "Mock drift-state fields",
        [ROOT / "cnn_fpga" / "benchmark" / "run_hil_suite.py"],
        "The mock noise provider emits sigma, mu_q, mu_p, theta_deg and metadata.signal_type for synthetic/effective drift probes.",
        "This is not board-generated drift evidence and does not validate real-device distributions.",
    )

    expected_fig02_columns = {"scenario", "mode", "final_ler_mean", "final_ler_sd", "n_repeats"}
    fig02_columns = set(fig02[0].keys()) if fig02 else set()
    all_repeats_two = bool(fig02) and all(row.get("n_repeats") == "2" for row in fig02)
    add_check(
        checks,
        "SB-A08",
        expected_fig02_columns.issubset(fig02_columns) and all_repeats_two,
        "Source-data statistical boundary",
        [FIG_DIR / "source_data_fig02_main_results.csv"],
        "Fig. 2 source data carries final_ler_mean, final_ler_sd and n_repeats=2 for every checked row.",
        "The SD column remains descriptive; this does not provide CI, p-value or stronger statistical evidence.",
    )

    add_check(
        checks,
        "SB-W01",
        True,
        "Known limitation",
        [
            ROOT / "physics" / "syndrome_measurement.py",
            ROOT / "physics" / "logical_tracking.py",
            ROOT / "cnn_fpga" / "runtime" / "fast_loop_emulator.py",
        ],
        "The checked anchors do not constitute a complete finite-energy GKP physical model, wrapped-posterior decoder, holdout drift protocol, or real-board validation.",
        "Keep manuscript language at approximate-model and fixed software-HIL protocol level.",
        warn=True,
    )

    fail_count = sum(1 for check in checks if check.status == "FAIL")
    warn_count = sum(1 for check in checks if check.status == "WARN")
    pass_count = sum(1 for check in checks if check.status == "PASS")

    status = "FAIL" if fail_count else ("PASS_WITH_LIMITATIONS" if warn_count else "PASS")
    return {
        "status": status,
        "scope": "submission_draft_symbol_boundary_audit_v1",
        "date": date.today().isoformat(),
        "target_manuscript": rel(TEX_PATH),
        "boundary": (
            "Read-only symbol/code/source-data boundary check for the submission draft. "
            "It does not run experiments, validate a complete physical model, prove "
            "statistical significance, validate hardware, or change any governance task state."
        ),
        "counts": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "checks": [check.__dict__ for check in checks],
    }


def table_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_reports(result: dict[str, object]) -> None:
    REPORT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 投稿稿符号边界机械审计报告",
        "",
        f"日期：{result['date']}",
        "",
        f"对象：`{result['target_manuscript']}`",
        "",
        "## 作用域",
        "",
        "本文档由 `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_symbol_boundary.py` 生成。它只做投稿稿符号、源码字段和 source-data 统计边界的机械一致性检查。",
        "",
        "本文档不是新实验，不运行 benchmark，不导入 runtime，不证明完整物理噪声模型，不补 CI / p-value，不验证 real-board，也不改变 `.tflite`、HIL、statcalib 或当前治理任务的证据等级。",
        "",
        "## 审计结论",
        "",
        f"- Status: `{result['status']}`",
        f"- Checks passed: `{result['counts']['pass']}`",
        f"- Warnings: `{result['counts']['warn']}`",
        f"- Checks failed: `{result['counts']['fail']}`",
        "- 主结论：当前投稿稿中的 `lambda=sqrt(2pi)`、`final_ler` definition box、wrap 区间、半晶格逻辑边界、fast-loop 指标、mock drift 字段和 Fig. 2 descriptive-SD 口径都有可定位的源码或 source-data 锚点。",
        "- 解释边界：这些锚点只支撑 fixed software-HIL protocol 下的可追溯写法；不能写成真实硬件 logical error rate、完整 finite-energy GKP 模型、holdout generalization 或显著性结论。",
        "",
        "## 检查明细",
        "",
        "| ID | Status | Topic | Paths | Detail | Boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for check in result["checks"]:
        lines.append(
            "| `{id}` | `{status}` | {topic} | {paths} | {detail} | {boundary} |".format(
                id=table_cell(check["check_id"]),
                status=table_cell(check["status"]),
                topic=table_cell(check["topic"]),
                paths=table_cell(", ".join(f"`{path}`" for path in check["paths"])),
                detail=table_cell(check["detail"]),
                boundary=table_cell(check["boundary"]),
            )
        )

    lines.extend(
        [
            "",
            "## 可写入稿件的保守结论",
            "",
            "- 可以写：the implementation shares the GKP lattice constant across syndrome wrapping, logical-boundary scoring and fast-loop limits.",
            "- 可以写：`final_ler` is a protocol-defined logical-error proxy produced by the current software-HIL emulator.",
            "- 可以写：the mock provider exposes an effective drift state through `sigma`, `mu_q`, `mu_p` and `theta_deg`.",
            "- 不能写：the results are hardware logical-error measurements, complete physical-noise validation, statistically significant improvements, or generalization across realistic device drift.",
            "",
            "## 后续投稿前仍需补的材料",
            "",
            "1. 将 `final_ler` definition box 与 `LogicalErrorTracker` 行为纳入轻量单元测试或 CI preflight。",
            "2. 补 oracle/known-noise affine baseline 或 wrapped-Gaussian / nearest-lattice baseline。",
            "3. 补 unseen drift family 或 holdout protocol。",
            "4. 把主结果 repeats 从 `2` 提高到可报告 paired CI 的水平。",
            "5. 若保留硬件叙事，补 real-board source data、bitstream/RTL provenance、DMA trace、latency/resource report 和 failure logs；否则继续保留 placeholder。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = audit()
    write_reports(result)
    print(json.dumps({"status": result["status"], "report": str(REPORT_MD)}, ensure_ascii=False))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
