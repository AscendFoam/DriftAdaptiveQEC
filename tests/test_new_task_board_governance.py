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
    assert statuses["T6.26.3"] == "In Progress"
    assert all(
        statuses[task] == "Todo"
        for task in expected_tasks
        - {"T6.20.1", "T6.20.2", "T6.20.3", "T6.20.4", "T6.25.1", "T6.25.2", "T6.25.3", "T6.25.4", "T6.26.3"}
        - dropped
    )
    assert statuses["T6.9.2"] == "Blocked"
    assert statuses["T7.3.1"] == "Blocked"

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
