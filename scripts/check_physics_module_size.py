"""Enforce the physics module size policy."""

from __future__ import annotations

from pathlib import Path


MAX_LINES = 1000


def validate(physics_dir: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(physics_dir.rglob("*.py")):
        payload = path.read_bytes()
        line_count = len(payload.splitlines())
        if line_count > MAX_LINES:
            failures.append(
                f"{path}: {line_count} lines exceeds the {MAX_LINES}-line policy"
            )
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = validate(root / "physics")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"physics module size policy: PASS (max={MAX_LINES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
