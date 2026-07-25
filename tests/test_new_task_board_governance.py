from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "docs" / "new_task_board.md"
TASK_ROW = re.compile(
    r"^\|\s*(T(?:-RISK-\d{8}-\d+|\d+(?:\.\d+)+))\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE,
)


def _authoritative_task_statuses(board: str) -> dict[str, str]:
    """Return the first task-table occurrence, excluding later progress-log mentions."""

    statuses: dict[str, str] = {}
    for task_id, status in TASK_ROW.findall(board):
        statuses.setdefault(task_id, status.strip())
    return statuses


def test_new_board_sources_and_insertion_rule_are_live() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    for relative_path in (
        "docs/rough_plan.md",
        "docs/experiment_plan.md",
        "docs/new_risks.md",
    ):
        assert (ROOT / relative_path).is_file(), relative_path

    insertion_rule = board.split("## 插入任务区", 1)[1].split("## 进度日志", 1)[0]
    assert "`docs/new_risks.md`" in insertion_rule
    assert "`docs/risks.md`" not in insertion_rule

    experiment_plan = (ROOT / "docs" / "experiment_plan.md").read_text(
        encoding="utf-8"
    )
    for section in ("## 14.1", "## 15.1", "## 16.1", "# 17.", "## 17.1"):
        assert section in experiment_plan
    assert "`docs/new_task_board.md` 是本修订后的唯一当前执行顺序和状态源" in experiment_plan


def test_every_done_task_has_current_and_legacy_completion_records() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    statuses = _authoritative_task_statuses(board)
    assert statuses

    missing: list[tuple[str, str]] = []
    for task_id, status in statuses.items():
        if status != "Done":
            continue
        for directory in ("docs/new_tasks", "docs/tasks"):
            if not list((ROOT / directory).glob(f"{task_id}_*")):
                missing.append((task_id, directory))
    assert missing == []


def test_board_has_no_stale_cross_workspace_analysis_or_false_board_claim() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    assert "D:/Codes/Quantum/CNN_FPGA_GKP" not in board
    assert "尚未修改 [task_board.md]" not in board
    assert "## 任务改进说明(含参考文献)" not in board

    statuses = _authoritative_task_statuses(board)
    assert statuses["T6.1.1"] == "Blocked"
    assert statuses["T6.2.1"] == "Done"
    assert statuses["T6.2.2"] == "Done"
    assert statuses["T6.5.1"] == "Done"
    assert statuses["T6.5.2"] == "Done"
    assert statuses["T6.5.3"] == "Done"
    assert statuses["T6.6.1"] == "Done"
    assert statuses["T6.6.2"] == "Done"
    assert statuses["T6.6.3"] == "Done"
    assert statuses["T6.7.1"] == "Done"
    assert statuses["T6.7.2"] == "Done"
    assert statuses["T6.7.3"] == "Done"
    assert statuses["T6.7.4"] == "Done"
    assert statuses["T6.8.1"] == "Done"
    assert statuses["T6.8.2"] == "Done"
    assert statuses["T6.8.3"] == "Done"
    assert statuses["T6.8.4"] == "Done"
    assert statuses["T6.8.5"] == "Done"
    assert statuses["T6.8.6"] == "Done"
    assert statuses["T6.8.7"] == "Done"
    assert statuses["T6.9.1"] == "Done"
    assert statuses["T6.9.2"] == "Blocked"
    assert statuses["T6.9.3"] == "Done"
    assert statuses["T6.2.5"] == "Dropped"


def test_phase6_preboard_lane_is_not_globally_blocked_by_missing_board() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    current = re.search(r"^当前推荐任务：`([^`]+)`", board, re.MULTILINE)
    assert current is not None
    statuses = _authoritative_task_statuses(board)
    assert current.group(1) != "T6.9.2"
    assert statuses[current.group(1)] == "In Progress"

    phase6 = board.split("## Phase 6", 1)[1].split("## Phase 7", 1)[0]
    dependency_split = phase6.split("### Phase 6 依赖拆分", 1)[1].split(
        "### Milestone 6.1", 1
    )[0]
    assert "T6.2.1 -> T6.2.2" in dependency_split
    assert "不以 `T6.1.1` 为前置条件" in dependency_split
    assert "T6.1.1 -> T6.1.2 -> T6.1.3" in dependency_split
    assert "T6.2.3" in dependency_split
    assert "证据不互换" in dependency_split

    assert "10 个 family 各 100,000、聚合 1,000,000 cycles" in phase6
    for required_case in (
        "CRC/version",
        "stale/rollback/untrusted bank",
        "reset",
        "deadline",
        "commit race",
        "FIFO overflow/backpressure",
        "pause/drop/duplicate/reorder",
    ):
        assert required_case in phase6
    assert "T6.2.2 的软件长序列只作为回归基线，不得替代本项板上证据" in phase6
    assert "T6.2.2 的抽象故障只作预板回归和预期基线" in phase6

    production_report = json.loads(
        (ROOT / "docs/t6_2_1_production_rtl_audit.json").read_text(encoding="utf-8")
    )
    assert production_report["status"] == "PASS"
    assert production_report["mismatch_count"] == 0
    assert production_report["evidence_boundary"]["board_measured"] is False

    long_report = json.loads(
        (ROOT / "docs/t6_2_2_long_rtl_qualification.json").read_text(encoding="utf-8")
    )
    assert long_report["verdict"] == (
        "PASS_BOARD_INDEPENDENT_LONG_RTL_QUALIFICATION_READY_FOR_ROUTE_A"
    )
    assert long_report["aggregate_python"]["cycles"] >= 1_000_000
    assert sum(row["mismatches"] for row in long_report["cxxrtl_families"]) == 0
    assert long_report["aggregate_python"]["silent_overflow"] == 0
    assert "abstract" in long_report["transport_contract"]


def test_phase6a_route_a_contract_and_external_lanes_are_frozen() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    phase6a = board.split("## Phase 6A", 1)[1].split("## Phase 6B", 1)[0]
    statuses = _authoritative_task_statuses(board)

    expected_tasks = {
        "T6.5.1",
        "T6.5.2",
        "T6.5.3",
        "T6.6.1",
        "T6.6.2",
        "T6.6.3",
        "T6.7.1",
        "T6.7.2",
        "T6.7.3",
        "T6.7.4",
        "T6.8.1",
        "T6.8.2",
        "T6.8.3",
        "T6.8.4",
        "T6.8.5",
        "T6.8.6",
        "T6.8.7",
        "T6.9.1",
        "T6.9.2",
        "T6.9.3",
    }
    assert {task for task in statuses if task.startswith("T6.") and task in expected_tasks} == expected_tasks
    assert statuses["T6.5.1"] == "Done"
    assert statuses["T6.5.2"] == "Done"
    assert statuses["T6.5.3"] == "Done"
    assert statuses["T6.6.1"] == "Done"
    assert statuses["T6.6.2"] == "Done"
    assert statuses["T6.6.3"] == "Done"
    assert statuses["T6.7.1"] == "Done"
    assert statuses["T6.7.2"] == "Done"
    assert statuses["T6.7.3"] == "Done"
    assert statuses["T6.7.4"] == "Done"
    assert statuses["T6.8.1"] == "Done"
    assert statuses["T6.8.2"] == "Done"
    assert statuses["T6.8.3"] == "Done"
    assert statuses["T6.8.4"] == "Done"
    assert statuses["T6.8.5"] == "Done"
    assert statuses["T6.8.6"] == "Done"
    assert statuses["T6.8.7"] == "Done"
    assert statuses["T6.9.1"] == "Done"
    assert statuses["T6.9.2"] == "Blocked"
    assert statuses["T6.9.3"] == "Done"
    assert all(
        statuses[task] == "Todo"
        for task in expected_tasks - {"T6.5.1", "T6.5.2", "T6.5.3", "T6.6.1", "T6.6.2", "T6.6.3", "T6.7.1", "T6.7.2", "T6.7.3", "T6.7.4", "T6.8.1", "T6.8.2", "T6.8.3", "T6.8.4", "T6.8.5", "T6.8.6", "T6.8.7", "T6.9.1", "T6.9.2", "T6.9.3"}
    )

    for milestone in ("6.5", "6.6", "6.7", "6.8", "6.9"):
        assert f"### Milestone {milestone}" in phase6a
    for contract_term in (
        "同一 syndrome 输入",
        "MAP-LUT",
        "定点精度",
        "6-cycle event/action path",
        "versioned A/B bank",
        "update cadence",
        "observed-only",
        "wall-clock 与计算预算",
    ):
        assert contract_term in phase6a
    for comparator in (
        "standard binning",
        "static joint MAP",
        "Window MAP",
        "EWMA adaptive MAP",
        "Kalman adaptive MAP",
        "legacy CNN residual",
        "proposed Route-A",
        "hidden-state oracle",
    ):
        assert comparator in phase6a
    for scenario in (
        "mean、variance、correlation、periodic",
        "step、telegraph、burst、readout/reset、leakage、compound",
    ):
        assert scenario in phase6a

    assert "aggregate paired LER improvement 95% 下界必须 `>0`" in phase6a
    assert "`55/512 > 37/512`" in phase6a
    assert "聚合不少于 `1e6` cycles" in phase6a
    assert "零 bit mismatch、零 undefined action、零 silent overflow" in phase6a
    assert "https://github.com/Matteo-Puviani/GQF" in phase6a
    assert "禁止“超过 Puviani NMF”" in phase6a
    assert "禁止“比已有 FPGA decoder 更快”" in phase6a
    assert "三条外部比较 lane 不混排" in phase6a
    assert "T6.9.2" in phase6a and "依赖 T6.2.3、T6.4 和 T6.9.1" in phase6a

    phase7_preamble = board.split("## Phase 7", 1)[1].split("### Milestone 7.1", 1)[0]
    assert "T6.9.3 是 V4 的历史 evidence snapshot" in phase7_preamble
    assert "`NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY`" in phase7_preamble
    assert "`T6.15.5=GO_SIM_PREBOARD`" in phase7_preamble
    assert "`T6.19.3=PASS_AUX_COMPARISON_INTEGRITY`" in phase7_preamble
    assert "不能据此挽救或重开主论文门" in phase7_preamble
    assert "所有 measured hardware 图表和措辞继续依赖 Blocked 的 T6.9.2" in phase7_preamble


def test_phase6b_predictive_risk_aware_v5_is_software_first_and_fail_closed() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    phase6b = board.split("## Phase 6B", 1)[1].split("## Phase 6C", 1)[0]
    statuses = _authoritative_task_statuses(board)

    expected_tasks = {
        *(f"T6.10.{i}" for i in range(1, 4)),
        *(f"T6.11.{i}" for i in range(1, 5)),
        *(f"T6.12.{i}" for i in range(1, 5)),
        *(f"T6.13.{i}" for i in range(1, 4)),
        *(f"T6.14.{i}" for i in range(1, 4)),
        *(f"T6.15.{i}" for i in range(1, 6)),
    }
    assert len(expected_tasks) == 22
    assert statuses["T6.10.1"] == "Done"
    assert statuses["T6.15.5"] == "Done"
    assert all(
        statuses[task] == "Dropped"
        for task in expected_tasks - {"T6.10.1", "T6.15.5"}
    )
    assert statuses["T6.9.2"] == "Blocked"

    for milestone in ("6.10", "6.11", "6.12", "6.13", "6.14", "6.15"):
        assert f"### Milestone {milestone}" in phase6b

    for required in (
        "旧结果只作诊断",
        "posterior-mixture/action-space",
        "strict-causal selector",
        "multiscale wrapped/circular feature extractor",
        "continuous-state static/trend/harmonic IMM",
        "BOCPD/telegraph",
        "activation-horizon posterior prediction",
        "uncertainty-marginalized posterior-predictive MAP-LUT compiler",
        "calibrated LER/CVaR risk gate",
        "真实 two-bank residency",
        "全新的 train/calibration/pilot/formal 四分割",
        "untouched unseen smooth-drift formal matrix",
        "worst-window endpoint 至少下降 `50%`",
        "min_b((p_L^b-p_L^V5)/p_L^b) >= 10%",
        "零 bit mismatch、零 undefined action、零 silent overflow",
        "source-to-action 恰为 6 cycles",
        "II=1、无 bubble",
        "actual parameterized production module/source hash",
        "GO_SIM_PREBOARD",
    ):
        assert required in phase6b

    assert "不依赖 Blocked 的 T6.9.2" in phase6b
    assert "T6.9.2 继续独立 Blocked" in phase6b
    assert "未通过它之前禁止 measured latency" in phase6b
    assert "若需 V6，必须另建新 protocol/split" in phase6b
    assert "strict-causal selector 相对 strongest fold-selected Window 为 `-0.2322%`" in phase6b
    assert "纯 action-space 增量只有 9 errors=`0.02549%`" in phase6b
    assert "NO_GO_V5_EARLY_HEADROOM_STOP" in phase6b

    authoritative = board.split("## 进度日志", 1)[0]
    task_ids = [task_id for task_id, _ in TASK_ROW.findall(authoritative)]
    duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
    assert duplicates == []


def test_phase6c_secondary_comparisons_are_post_phase6b_nonmixed_and_read_only() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    phase6c = board.split("## Phase 6C", 1)[1].split("## Phase 7", 1)[0]
    statuses = _authoritative_task_statuses(board)

    ordered_tasks = tuple(
        task
        for milestone in range(16, 20)
        for task in (f"T6.{milestone}.1", f"T6.{milestone}.2", f"T6.{milestone}.3")
    )
    expected_tasks = set(ordered_tasks)
    assert len(expected_tasks) == 12
    phase_states = [statuses[task] for task in ordered_tasks]
    assert set(phase_states) <= {"Done", "In Progress", "Todo"}
    assert phase_states.count("In Progress") <= 1
    # Sequential governance: a completed prefix, at most one active task, then
    # an untouched suffix.  This remains valid throughout Phase 6C instead of
    # pinning the test to its initial all-Todo snapshot.
    assert phase_states == (
        ["Done"] * phase_states.count("Done")
        + ["In Progress"] * phase_states.count("In Progress")
        + ["Todo"] * phase_states.count("Todo")
    )
    assert statuses["T6.9.2"] == "Blocked"

    for milestone in ("6.16", "6.17", "6.18", "6.19"):
        assert f"### Milestone {milestone}" in phase6c

    for required in (
        "严格在原 Phase 6B 完成后执行",
        "`T6.15.5=Done`",
        "六条证据 lane",
        "禁止 global leaderboard",
        "N/A` 表示不适用",
        "LITERATURE_ONLY",
        "OFFICIAL_CODE_REPRODUCTION",
        "PROJECT_NATIVE_MATCHED",
        "不进入 V5 `>=10%` LER 的 denominator",
        "single-mode square/isotropic Euclidean CPD 与 CI 的等价边界",
        "两 GKP-qubit error-corrected CNOT",
        "`9.9 dB` full surface–GKP threshold",
        "Direct NN、causal adaptive NN 与 RL controller",
        "共同 wall-clock",
        "amazon-science/LatticeAlgorithms.jl",
        "NOT_RUN_SCOPE_GATE",
        "same-task comparator",
        "PASS_AUX_COMPARISON_INTEGRITY",
        "即使所有结果为负或不可比也可通过",
    ):
        assert required in phase6c

    assert "本阶段不依赖 Blocked 的 T6.9.2" in phase6c
    assert "硬件 measured 字段继续为 null" in phase6c
    assert "不能升级 T6.15.5 或 T6.9.2" in phase6c

    insertion = board.split("## 插入任务区", 1)[1].split("## 进度日志", 1)[0]
    assert "T-RISK-20260720-02" in insertion
    assert "Milestone 6.16—6.19 共 12 个 Todo task" in insertion


def test_phase6d_dual_evidence_lanes_are_strong_baseline_first_and_nontransferable() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    phase6d = board.split("## Phase 6D", 1)[1].split("## Phase 7", 1)[0]
    statuses = _authoritative_task_statuses(board)

    expected_tasks = {
        *(f"T6.20.{i}" for i in range(1, 5)),
        *(f"T6.21.{i}" for i in range(1, 5)),
        *(f"T6.22.{i}" for i in range(1, 5)),
        *(f"T6.23.{i}" for i in range(1, 6)),
        *(f"T6.24.{i}" for i in range(1, 6)),
        *(f"T6.25.{i}" for i in range(1, 5)),
        *(f"T6.26.{i}" for i in range(1, 5)),
    }
    assert len(expected_tasks) == 30
    assert statuses["T6.20.1"] == "Done"
    assert statuses["T6.20.2"] == "Done"
    assert statuses["T6.20.3"] == "Done"
    assert statuses["T6.20.4"] == "Done"
    dropped = {
        *(f"T6.21.{i}" for i in range(1, 5)),
        *(f"T6.22.{i}" for i in range(1, 5)),
        *(f"T6.23.{i}" for i in range(1, 6)),
        *(f"T6.24.{i}" for i in range(1, 6)),
        "T6.26.1",
        "T6.26.2",
    }
    assert all(statuses[task] == "Dropped" for task in dropped)
    assert statuses["T6.25.1"] == "Done"
    assert statuses["T6.25.2"] == "Done"
    assert statuses["T6.25.3"] == "Done"
    assert statuses["T6.25.4"] == "Done"
    assert statuses["T6.26.3"] == "Done"
    assert statuses["T6.26.4"] == "Done"
    assert all(
        statuses[task] == "Todo"
        for task in expected_tasks
        - {"T6.20.1", "T6.20.2", "T6.20.3", "T6.20.4", "T6.25.1", "T6.25.2", "T6.25.3", "T6.25.4", "T6.26.3", "T6.26.4"}
        - dropped
    )
    assert statuses["T6.9.2"] == "Blocked"
    assert statuses["T7.3.1"] == "Blocked"
    assert statuses["T7.1.5"] == "Done"
    assert statuses["T7.2.6"] == "Done"
    assert statuses["T7.3.2"] == "Done"
    assert statuses["T7.3.3"] == "Done"
    assert statuses["T7.3.4"] == "Done"
    assert statuses["T7.3.5"] == "Done"

    for milestone in ("6.20", "6.21", "6.22", "6.23", "6.24", "6.25", "6.26"):
        assert f"### Milestone {milestone}" in phase6d

    for required in (
        "multimode surface-square GKP",
        "periodic analog-MWPM",
        "exact logical-coset MLD",
        "K-MWM",
        "static-mixture exact MLD",
        "adapted SMC-EAP",
        "BOCPD/IMM",
        "true_metric_CPD_reference",
        "true_theta_exact_MLD_oracle",
        "全新 train/calibration/pilot/formal 四分割",
        "T6.18.3 的 seeds",
        "point `>=15%`",
        "paired 95% LCB `>=12%`",
        "posterior-predictive exact logical-coset MLD",
        "relative LER improvement 的 simultaneous paired 95% LCB 均须 `>10%`",
        "source-to-action 恰 6 cycles",
        "GO_TWO_LANE",
        "GO_MULTIMODE_ONLY",
        "GO_RTL_ONLY",
        "CNN/student",
        "不做加权总分",
    ):
        assert required in phase6d

    assert "T6.9.2 真板任务继续独立 Blocked" in phase6d
    assert "该 lane 不声称执行 multimode exact/posterior-predictive MLD" in phase6d
    assert "single-mode 六周期不证明 multimode 方法的硬件 latency" in board
    assert "multimode LER 不证明现有 RTL 执行 multimode decoder" in board
    assert "NO_GO_MULTIMODE_CAUSAL_HEADROOM" in phase6d
    assert "79,872" in phase6d
    assert "pilot/formal 保持未访问" in board
    assert (ROOT / "docs" / "phase6d_multimode_v1_cancellation_ledger.md").is_file()
    assert (ROOT / "docs" / "t6_25_1_single_mode_rtl_boundary_audit.json").is_file()
    assert (ROOT / "docs" / "t6_25_2_converged_rtl_formal.json").is_file()
    assert "converged production top" in phase6d

    registry = ROOT / "docs" / "multimode_strong_baseline_registry.md"
    assert registry.is_file()
    registry_text = registry.read_text(encoding="utf-8")
    for required in (
        "DIRECT_OFFICIAL_REPRODUCTION",
        "DIRECT_OFFICIAL_PENDING_REPRODUCTION",
        "paper-inspired adapted baseline",
        "不同 task signature",
        "true_metric_CPD_reference",
        "true_theta_exact_MLD_oracle",
        "在冻结 benchmark 上达到 SOTA",
    ):
        assert required in registry_text

    insertion = board.split("## 插入任务区", 1)[1].split("## 进度日志", 1)[0]
    assert "T-RISK-20260721-01" in insertion
    assert "Milestone 6.20—6.26 共 30 个 task" in insertion


def test_phase9_performance_first_single_mode_reboot_is_nonblocking_and_fail_closed() -> None:
    board = BOARD_PATH.read_text(encoding="utf-8")
    phase9 = board.split("## Phase 9", 1)[1].split("## 与规划约束", 1)[0]
    statuses = _authoritative_task_statuses(board)

    expected_tasks = {
        *(f"T9.1.{i}" for i in range(1, 6)),
        *(f"T9.2.{i}" for i in range(1, 8)),
        *(f"T9.3.{i}" for i in range(1, 5)),
        *(f"T9.4.{i}" for i in range(1, 6)),
        *(f"T9.5.{i}" for i in range(1, 5)),
        *(f"T9.6.{i}" for i in range(1, 6)),
        *(f"T9.7.{i}" for i in range(1, 5)),
        *(f"T9.8.{i}" for i in range(1, 4)),
    }
    assert len(expected_tasks) == 37
    actual_phase9_tasks = {task_id for task_id, _ in TASK_ROW.findall(phase9)}
    assert actual_phase9_tasks == expected_tasks
    blocked = {"T9.1.2", "T9.7.3", "T9.7.4"}
    assert all(statuses[task] == "Blocked" for task in blocked)
    assert statuses["T9.1.1"] == "Done"
    assert statuses["T9.1.3"] == "Done"
    assert statuses["T9.1.4"] == "Done"
    assert statuses["T9.1.5"] == "Done"
    assert statuses["T9.2.1"] == "Done"
    assert statuses["T9.2.2"] == "Done"
    assert statuses["T9.2.3"] == "In Progress"
    assert all(
        statuses[task] == "Todo"
        for task in expected_tasks
        - blocked
        - {
            "T9.1.1",
            "T9.1.3",
            "T9.1.4",
            "T9.1.5",
            "T9.2.1",
            "T9.2.2",
            "T9.2.3",
        }
    )
    assert statuses["T7.3.5"] == "Done"
    assert "当前推荐任务：`T9.2.3`" in board

    for milestone in ("9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"):
        assert f"### Milestone {milestone}" in phase9

    for required in (
        "BLOCKED_OFFICIAL_EXACT_ASSETS",
        "PAPER_CONSTRAINED_REIMPLEMENTATION",
        "20 个 paired roots / 40 agents",
        "不依赖 `T9.1.2`",
        "QUALIFIED_PAPER_CONSTRAINED_BASELINE",
        "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION",
        "52/52 gates",
        "81/81 mutation",
        "complex raw/recorded IQ",
        "不得复用 backend A 的 transition kernel",
        "未参与训练的 exact backend",
        "trusted recovery codebook",
        "CNN/GRU、TCN、SSM 与 causal Transformer candidates",
        "hidden-state teacher",
        "composite key",
        "action/next-state",
        "base lane residual",
        "TBPTT",
        "跨 chunk carry",
        "latent identifiability",
        "predictive-risk representation",
        "pinned CPU core/thread",
        "slow inference/transfer 永不进入逐周期 critical path",
        "每 trajectory 至少运行到预注册 `10^4` cycles",
        "relative improvement point `>=15%`",
        "simultaneous paired 95% LCB `>=10%`",
        "information representation × history × architecture × selector/safety × codebook",
        "registered/project-native 状态",
        "raw/recorded IQ -> discriminator -> action -> trigger HIL",
        "raw/recorded-IQ matched-filter + discriminator",
        "conservative representative action probes",
        "`6-cycle/II=1` 只绑定 `discriminator-out -> action`",
        "PREBOARD_SYNTHETIC_IQ",
        "BOARD_RECORDED_IQ_REPLAY",
        "BOARD_LIVE_RAW_IQ_HIL",
        "EXTERNAL_SAME_TASK_MEASURED_SPEED",
        "GO_LER_REGISTERED_BEST/GO_LER_EXTERNAL_SOTA",
        "GO_LIFETIME_PROJECT_NATIVE/GO_LIFETIME_EXTERNAL_SOTA/GO_PHYSICAL_LIFETIME",
        "`Done` 或 terminal `Blocked/null` ledger",
        "paper_scope_verdict",
        "three_metric_sota_verdict",
        "绝不自动蕴含",
        "GO_SINGLE_PAPER",
        "GO_SPLIT_ALGORITHM_HARDWARE",
        "不得把 v1 legacy candidate label、三项胜场或加权总分写成 SOTA",
    ):
        assert required in phase9

    assert "`T9.1.2` 局部 `Blocked`" in phase9
    assert all(task in phase9 for task in ("T9.7.3", "T9.7.4"))
    assert "CXXRTL/P&R/host loop 代替" in phase9
    assert "Phase 8 的真实 GKP/QPU 接入" in phase9
    assert "immutable `analysis_sha256`" in phase9

    task_rows: dict[str, list[str]] = {}
    for line in phase9.splitlines():
        if not line.startswith("| T9."):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 5:
            task_rows[cells[0]] = cells
    parent_protocol = json.loads(
        (ROOT / "docs" / "t9_1_1_three_lane_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_projection = parent_protocol["artifact_registry"][
        "task_board"
    ]["payload"]
    live_projection = {
        task_id: {
            "task_id": task_id,
            "task": task_rows[task_id][2],
            "source": task_rows[task_id][4],
        }
        for task_id in frozen_projection
    }
    assert live_projection == frozen_projection
    assert "codebook actions" not in task_rows["T9.2.4"][3]
    assert "conservative representative action probes" in task_rows["T9.2.4"][3]
    assert "T9.1.3" not in task_rows["T9.3.2"][4]
    assert "T9.1.5" in task_rows["T9.4.5"][4]
    assert "必须直接消费 T9.1.5" in task_rows["T9.6.1"][3]
    assert "必须直接消费 T9.1.5" in task_rows["T9.6.5"][3]
    assert "必须直接消费 T9.1.5" in task_rows["T9.8.1"][3]
    assert "不是 hard prerequisite" in task_rows["T9.1.4"][3]
    # These source cells are part of the immutable T9.1.1 semantic projection.
    assert task_rows["T9.1.3"][4] == (
        "`T9.1.1`, `R-N162`, `experiment_plan.md §20.2`"
    )
    assert task_rows["T9.1.4"][4] == (
        "`T9.1.1`, `T9.1.3`, `R-N164`, `experiment_plan.md §20.2`"
    )
    assert task_rows["T9.6.1"][4] == (
        "`T9.1.1`, `T9.4.5`, `T9.5.4`, `R-N168`"
    )
    assert task_rows["T9.6.5"][4] == (
        "`T9.2.4`, `T9.6.2--T9.6.4`, `R-N162--R-N166`, `R-N168`"
    )
    assert task_rows["T9.8.1"][4] == "`T9.6.5`, `T9.7.4`, `R-N168`"
    assert "Puviani official exact/surpass 保持 null" in task_rows["T9.6.5"][3]
    assert "仅在独立 Phase 8 QPU/real-GKP 证据缺失时保持 null" in task_rows["T9.6.5"][3]
    assert "T6.9.2" not in task_rows["T9.7.3"][4]
    assert "T6.9.2 仅可作为" in task_rows["T9.7.3"][3]
    assert "terminal `Blocked/null` ledger" in task_rows["T9.8.1"][3]

    protocol_path = ROOT / "docs" / "t9_1_1_three_lane_protocol.json"
    protocol_source = ROOT / "docs" / "t9_1_1_three_lane_protocol_source_data.csv"
    protocol_markdown = ROOT / "docs" / "phase9_three_lane_protocol.md"
    assert protocol_path.is_file() and protocol_source.is_file() and protocol_markdown.is_file()
    protocol_report = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol_report["verdict"] == "PASS_PHASE9_THREE_INDEPENDENT_LANE_PROTOCOL_FROZEN"
    assert protocol_report["gate_summary"] == {"passed": 36, "failed": []}
    assert all(row["current_result"]["result_verdict"] is None for row in protocol_report["lanes"])
    assert protocol_report["external_claim_slots"]["PUVIANI_NMF_SURPASS"]["value"] is None
    assert protocol_report["external_claim_slots"]["PHYSICAL_BREAK_EVEN"]["value"] is None

    experiment_plan = (ROOT / "docs" / "experiment_plan.md").read_text(
        encoding="utf-8"
    )
    section20 = experiment_plan.split("# 20.", 1)[1].split("\n[1]:", 1)[0]
    for required in (
        "Puviani 资产缺失的非阻塞证据合同",
        "OFFICIAL_EXACT_REPRODUCTION",
        "PAPER_CONSTRAINED_REIMPLEMENTATION",
        "双后端、action-conditioned 数字孪生",
        "同权限模型 tournament",
        "六态 formal 与高速板 HIL 门",
        "GO_LER_SOTA",
        "GO_LIFETIME",
        "GO_HIL_SPEED",
        "## 20.9",
        "POST_OUTCOME_GOVERNANCE_ADDENDUM_NO_RETROACTIVE_RESEAL",
        "有限 total recurrence",
        "long-history、identifiability 与 matched compute",
        "factorial benchmark 与四级 HIL",
        "paper_scope_verdict",
        "three_metric_sota_verdict",
    ):
        assert required in section20

    risks = (ROOT / "docs" / "new_risks.md").read_text(encoding="utf-8")
    for risk_number in range(162, 181):
        assert f"R-N{risk_number}" in risks
    assert "| R-N168 | Mitigated |" in risks
    assert "| R-N170 | Open |" in risks
    assert "| R-N171 | Mitigated |" in risks
    assert "| R-N172 | Mitigated |" in risks
    assert "| R-N173 | Mitigated |" in risks
    assert "| R-N174 | Open |" in risks
    assert "| R-N175 | Mitigated |" in risks
    assert "| R-N176 | Mitigated |" in risks
    assert "| R-N177 | Mitigated |" in risks
    assert "| R-N178 | Open |" in risks
    assert "| R-N179 | Open |" in risks
    assert "| R-N180 | Open |" in risks
    assert "| 2026-07-24 | T9.1.3 完成与反简化复核 | 不插入 |" in risks
    assert "| 2026-07-25 | T9.1.4 完成与反简化复核 | 不插入 |" in risks
    assert "| 2026-07-25 | T9.1.5 完成与反简化复核 | 不插入 |" in risks
    assert "| 2026-07-26 | T9.2.1 完成与反简化复核 | 不插入 |" in risks
    assert "| 2026-07-26 | T9.2.2 完成与反简化复核 | 不插入 |" in risks

    canonical_record = (
        ROOT
        / "docs"
        / "new_tasks"
        / "T9.1.3_puviani_paper_constrained_artifacts.md"
    )
    mirror_record = (
        ROOT / "docs" / "tasks" / "T9.1.3_puviani_paper_constrained_artifacts.md"
    )
    assert canonical_record.is_file()
    assert mirror_record.is_file()
    canonical_text = canonical_record.read_text(encoding="utf-8")
    assert "QUALIFIED_PAPER_CONSTRAINED_BASELINE" in canonical_text
    assert "52/52 gates、81/81 mutations" in canonical_text
    assert "official exact、Puviani surpass" in canonical_text
    assert "保持 typed null" in canonical_text

    t914_canonical = (
        ROOT
        / "docs"
        / "new_tasks"
        / "T9.1.4_phase9_baseline_search_power_registry.md"
    )
    t914_mirror = (
        ROOT
        / "docs"
        / "tasks"
        / "T9.1.4_phase9_baseline_search_power_registry.md"
    )
    assert t914_canonical.is_file()
    assert t914_mirror.is_file()
    t914_text = t914_canonical.read_text(encoding="utf-8")
    for required in (
        "31 个方法",
        "18 个 mandatory",
        "required `N=806`",
        "planned `N=808/backend=8×101`",
        "lifetime 功效",
        "23 个 bibliographic raw hits",
        "36/36",
        "46 passed",
        "external_sota",
        "全部保持 null",
    ):
        assert required in t914_text

    t915_canonical = (
        ROOT
        / "docs"
        / "new_tasks"
        / "T9.1.5_phase9_scoped_claim_amendment.md"
    )
    t915_mirror = (
        ROOT
        / "docs"
        / "tasks"
        / "T9.1.5_phase9_scoped_claim_amendment.md"
    )
    assert t915_canonical.is_file()
    assert t915_mirror.is_file()
    t915_text = t915_canonical.read_text(encoding="utf-8")
    for required in (
        "29 个 lane-qualified legacy outputs",
        "9 个 scoped states",
        "17 条 forbidden transfers",
        "27 条 synthetic",
        "127 rows",
        "36/36 gates",
        "36/36 targeted mutations",
        "全部保持 typed null",
    ):
        assert required in t915_text

    t915_report_path = (
        ROOT / "docs" / "t9_1_5_scoped_claim_amendment.json"
    )
    t915_source_path = (
        ROOT
        / "docs"
        / "t9_1_5_scoped_claim_amendment_source_data.csv"
    )
    t915_markdown_path = (
        ROOT / "docs" / "phase9_scoped_claim_amendment.md"
    )
    t915_config_path = (
        ROOT
        / "configs"
        / "phase9"
        / "t9_1_5_scoped_claim_amendment.json"
    )
    t915_release_pin_path = (
        ROOT
        / "configs"
        / "phase9"
        / "t9_1_5_release_pin.json"
    )
    assert all(
        path.is_file()
        for path in (
            t915_report_path,
            t915_source_path,
            t915_markdown_path,
            t915_config_path,
            t915_release_pin_path,
        )
    )
    t915_report = json.loads(
        t915_report_path.read_text(encoding="utf-8")
    )
    assert (
        t915_report["verdict"]
        == "PASS_T9_1_5_PARENT_BOUND_SCOPED_CLAIM_AMENDMENT"
    )
    assert t915_report["gate_summary"] == {
        "passed": 36,
        "total": 36,
        "failed": [],
    }
    assert t915_report["semantic_mutation_audit"]["count"] == 36
    assert t915_report["semantic_mutation_audit"]["detected"] == 36
    assert len(t915_report["legacy_migration_table"]) == 29
    assert len(t915_report["state_definitions"]) == 9
    assert len(t915_report["forbidden_transfers"]) == 17
    assert len(t915_report["predicate_fixtures"]) == 27
    assert len(t915_report["revocation_fixtures"]) == 4
    assert t915_report["source_data"]["rows"] == 127
    t915_release_pin = json.loads(
        t915_release_pin_path.read_text(encoding="utf-8")
    )
    assert t915_release_pin["analysis_sha256"] == t915_report[
        "analysis_sha256"
    ]
    assert set(t915_report["current_claim_state"].values()) == {None}
    assert t915_report["performance_state"]["protocol_only"] is True

    t921_canonical = (
        ROOT
        / "docs"
        / "new_tasks"
        / "T9.2.1_phase9_causal_twin_contract.md"
    )
    t921_mirror = (
        ROOT
        / "docs"
        / "tasks"
        / "T9.2.1_phase9_causal_twin_contract.md"
    )
    assert t921_canonical.is_file() and t921_mirror.is_file()
    assert t921_canonical.read_bytes() == t921_mirror.read_bytes()
    t921_text = t921_canonical.read_text(encoding="utf-8")
    for required in (
        "PASS_T9_2_1_CAUSAL_TWIN_CONTRACT_FROZEN",
        "78775658b4c9fa3a768252e2d529e39a0a90924c28158de2e4a527ba3052fe34",
        "1,024/1,024 nominal cells",
        "131,072/131,072 transition cells",
        "196,608/196,608 composition quotient cells",
        "16,777,216",
        "40/40 machine gates",
        "40/40 one-gate-one-mutation",
        "62 passed",
        "previous_composite_key",
        "typed `null`",
    ):
        assert required in t921_text

    t921_report_path = (
        ROOT / "docs" / "t9_2_1_causal_twin_contract.json"
    )
    t921_manifest_path = (
        ROOT / "docs" / "t9_2_1_causal_twin_totality_manifest.json"
    )
    t921_source_path = (
        ROOT / "docs" / "t9_2_1_causal_twin_contract_source_data.csv"
    )
    t921_markdown_path = (
        ROOT / "docs" / "phase9_causal_twin_contract.md"
    )
    t921_config_path = (
        ROOT / "configs" / "phase9" / "t9_2_1_causal_twin_contract.json"
    )
    t921_release_pin_path = (
        ROOT / "configs" / "phase9" / "t9_2_1_release_pin.json"
    )
    assert all(
        path.is_file()
        for path in (
            t921_report_path,
            t921_manifest_path,
            t921_source_path,
            t921_markdown_path,
            t921_config_path,
            t921_release_pin_path,
        )
    )
    t921_report = json.loads(
        t921_report_path.read_text(encoding="utf-8")
    )
    assert (
        t921_report["verdict"]
        == "PASS_T9_2_1_CAUSAL_TWIN_CONTRACT_FROZEN"
    )
    assert t921_report["gate_summary"] == {
        "passed": 40,
        "total": 40,
        "failed": [],
    }
    assert t921_report["semantic_mutation_audit"]["detected"] == 40
    assert t921_report["source_data"]["rows"] == 246
    assert all(
        value is None
        for fields in t921_report["current_null_state"].values()
        for value in fields.values()
    )

    t922_canonical = (
        ROOT
        / "docs"
        / "new_tasks"
        / "T9.2.2_phase9_backend_a.md"
    )
    t922_mirror = (
        ROOT
        / "docs"
        / "tasks"
        / "T9.2.2_phase9_backend_a.md"
    )
    assert t922_canonical.is_file() and t922_mirror.is_file()
    assert t922_canonical.read_bytes() == t922_mirror.read_bytes()
    t922_text = t922_canonical.read_text(encoding="utf-8")
    for required in (
        "PASS_T9_2_2_BACKEND_A_QUALIFIED",
        "95f11f0dc17a1799a97a69af6908765d1e00d75399ef3d07277d075ada74f624",
            "联合 oscillator–qutrit",
        "20/20 machine gates",
        "20/20 targeted semantic mutations",
        "27/27 core qualification checks",
        "63 passed",
        "raw trace",
        "typed `null`",
    ):
        assert required in t922_text

    t922_report_path = (
        ROOT / "docs" / "t9_2_2_backend_a_qualification.json"
    )
    t922_source_path = (
        ROOT / "docs" / "t9_2_2_backend_a_qualification_source_data.csv"
    )
    t922_markdown_path = (
        ROOT / "docs" / "phase9_backend_a_qualification.md"
    )
    t922_config_path = (
        ROOT / "configs" / "phase9" / "t9_2_2_backend_a.json"
    )
    t922_release_pin_path = (
        ROOT / "configs" / "phase9" / "t9_2_2_release_pin.json"
    )
    assert all(
        path.is_file()
        for path in (
            t922_report_path,
            t922_source_path,
            t922_markdown_path,
            t922_config_path,
            t922_release_pin_path,
        )
    )
    t922_report = json.loads(
        t922_report_path.read_text(encoding="utf-8")
    )
    assert (
        t922_report["verdict"]
        == "PASS_T9_2_2_BACKEND_A_QUALIFIED"
    )
    assert t922_report["gate_summary"] == {
        "passed": 20,
        "total": 20,
        "all_passed": True,
    }
    assert t922_report["mutation_summary"] == {
        "detected": 20,
        "total": 20,
        "all_detected": True,
    }
    assert len(t922_report["qualification"]["checks"]) == 27
    assert all(t922_report["qualification"]["checks"].values())
    assert all(
        value is None
        for value in t922_report["qualification"]["claim_state"].values()
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/phase9_baseline_search_power_registry.md" in readme
    assert "required 806、planned 808 clusters/backend" in readme
    assert "docs/phase9_scoped_claim_amendment.md" in readme
    assert "29 条 lane-qualified legacy migration" in readme
    assert "docs/phase9_causal_twin_contract.md" in readme
    assert "docs/phase9_backend_a_qualification.md" in readme

    addendum_path = (
        ROOT / "docs" / "t9_1_3_post_outcome_governance_addendum.json"
    )
    addendum_source = (
        ROOT / "docs" / "t9_1_3_post_outcome_governance_source_data.csv"
    )
    input_contract_path = (
        ROOT / "configs" / "phase9" / "t9_1_4_input_contract.json"
    )
    assert addendum_path.is_file()
    assert addendum_source.is_file()
    assert input_contract_path.is_file()
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert (
        addendum["schema_version"]
        == "t9.1.3-post-outcome-governance-addendum-v2"
    )
    assert (
        addendum["status"]
        == "POST_OUTCOME_GOVERNANCE_ADDENDUM_NO_RETROACTIVE_RESEAL"
    )
    assert (
        addendum["analysis_sha256"]
        == "27f8226e6658e0a5b2d4e9bd2d55798478deb590ac639d05cb4248a9cafe6e5c"
    )
    assert addendum["gate_summary"] == {
        "passed": 16,
        "total": 16,
        "failed": [],
    }
    assert addendum["semantic_mutation_audit"]["count"] == 14
    assert addendum["semantic_mutation_audit"]["detected"] == 14
    assert addendum["semantic_mutation_audit"]["all_detected"] is True
    assert addendum["terminal_branch"] == "QUALIFIED"
    assert addendum["current_terminal_resolution"][
        "resolved_terminal_state"
    ] == "QUALIFIED_PAPER_CONSTRAINED_BASELINE"
    assert addendum["current_terminal_resolution"]["releases_t9_1_4"] is True
    assert addendum["current_terminal_resolution"][
        "matched_phase9_ranking_eligible"
    ] is False
    assert set(addendum["claim_slots"].values()) == {None}
    assert addendum["ranking_boundary"]["sota_claim_eligible"] is False
    assert addendum["downstream_semantic_seal"][
        "raw_addendum_hash_required_by_T9_1_4"
    ] is False
    assert (
        addendum["downstream_semantic_seal"]["semantic_sha256"]
        == "12ac54175ac13b0e2c4682d462a8da49dc367f9db0a67da76cc2347db6f7cb22"
    )

    input_contract = json.loads(
        input_contract_path.read_text(encoding="utf-8")
    )
    assert (
        input_contract["schema_version"]
        == "t9.1.4-paper-constrained-input-contract-v2"
    )
    assert {
        row["terminal_state"]
        for row in input_contract["terminal_state_mapping"]
    } == {
        "QUALIFIED_PAPER_CONSTRAINED_BASELINE",
        "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION",
    }
    no_go_fixture = input_contract["no_go_failure_manifest_fixture"]
    assert (
        no_go_fixture["terminal_result"]
        == "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION"
    )
    assert set(no_go_fixture["typed_null_payload"].values()) == {None}
    downstream_contract = input_contract[
        "downstream_addendum_semantic_contract"
    ]
    assert downstream_contract["accepted_addendum_schema_versions"] == [
        "t9.1.3-post-outcome-governance-addendum-v2"
    ]
    assert downstream_contract["accepted_terminal_branches"] == [
        "QUALIFIED",
        "NO_GO",
    ]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "第 14—20.9 节" in readme
    assert "40,000 training rows" in readme
    assert "`official_exact`、`puviani_surpass`、`paper_scale_lifetime`" in readme
    assert "physical/SOTA claim 不开放" in readme
    assert "matched/SOTA eligibility=false" in readme
    assert "三个实际 claim slot 保持 typed null" in readme

    insertion = board.split("## 插入任务区", 1)[1].split("## 进度日志", 1)[0]
    assert "T-RISK-20260722-01" in insertion
    assert "Milestone 9.1—9.8 共 34 个 task" in insertion
    assert "T-RISK-20260723-01" in insertion
    assert "总计 37 个 Phase 9 task" in insertion


def test_zotero_migration_bibliographies_retain_all_41_entries() -> None:
    task_dir = ROOT / "docs" / "tasks"
    paths = (
        task_dir / "T-RISK-20260706-01_zotero_supplement.bib",
        task_dir / "T-RISK-20260706-01_zotero_completion_round2.bib",
    )
    entry_counts = []
    for path in paths:
        assert path.is_file()
        entry_counts.append(
            sum(
                line.startswith("@")
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        )
    assert entry_counts == [6, 35]
    assert sum(entry_counts) == 41
