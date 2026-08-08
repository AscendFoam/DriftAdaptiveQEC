"""Content-addressed, no-double-write raw transaction for Phase-9 T04.

Large scientific payloads are written once into a same-volume staging area,
fsynced, hashed, atomically renamed to their SHA-256 address, reopened and
revalidated.  Only then may a small cell receipt be published.  Finalization
builds an inventory of object references; it never creates a monolithic ZIP or
a second merged copy of the raw rows.

An object without a receipt is an orphan and has no voting rights.  A receipt
without every valid object is rejected.  Existing receipts are immutable:
idempotent byte-identical replay is allowed, conflicting replay is fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


OBJECT_SCHEMA = "PHASE9-CONTENT-ADDRESSED-OBJECT-V1"
RECEIPT_SCHEMA = "PHASE9-POWERED-TWIN-CHUNK-RECEIPT-V1"
INVENTORY_SCHEMA = "PHASE9-POWERED-TWIN-OBJECT-INVENTORY-V1"
MANIFEST_SCHEMA = "PHASE9-POWERED-TWIN-EXECUTION-MANIFEST-V1"
ATTEMPT_SCHEMA = "PHASE9-POWERED-TWIN-ATTEMPT-EVENT-V1"
BUFFER_BYTES = 8 * 1024 * 1024
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "run_id",
        "config_sha256",
        "plan_sha256",
        "source_snapshot_sha256",
        "cell",
        "diagnostics",
        "runtime_fingerprint",
        "objects",
        "receipt_sha256",
    }
)
RUNTIME_FINGERPRINT_FIELDS = frozenset(
    {
        "runner_id",
        "python",
        "numpy",
        "scipy",
        "psutil",
        "platform",
        "thread_environment",
        "seed_namespace",
    }
)
THREAD_ENVIRONMENT_FIELDS = frozenset(
    {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(BUFFER_BYTES)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {label}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_runtime_fingerprint(
    fingerprint: object,
    *,
    runner_id: str,
    seed_namespace: str,
) -> Mapping[str, Any]:
    if not isinstance(fingerprint, Mapping):
        raise RuntimeError("receipt runtime fingerprint lineage mismatch")
    thread_environment = fingerprint.get("thread_environment")
    python_version = fingerprint.get("python")
    if (
        set(fingerprint) != RUNTIME_FINGERPRINT_FIELDS
        or fingerprint.get("runner_id") != runner_id
        or fingerprint.get("seed_namespace") != seed_namespace
        or not isinstance(python_version, list)
        or len(python_version) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in python_version
        )
        or not isinstance(fingerprint.get("numpy"), str)
        or not fingerprint.get("numpy")
        or not isinstance(fingerprint.get("scipy"), str)
        or not fingerprint.get("scipy")
        or not isinstance(fingerprint.get("psutil"), str)
        or not fingerprint.get("psutil")
        or not isinstance(fingerprint.get("platform"), str)
        or not fingerprint.get("platform")
        or not isinstance(thread_environment, Mapping)
        or set(thread_environment) != THREAD_ENVIRONMENT_FIELDS
        or any(value != "1" for value in thread_environment.values())
    ):
        raise RuntimeError("receipt runtime fingerprint lineage mismatch")
    return fingerprint


def _inside(path: Path, parent: Path, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes its immutable transaction root") from exc
    return resolved


def _regular_nonsymlink(path: Path, name: str) -> None:
    information = path.lstat()
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISREG(information.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")


def _fsync_file(path: Path) -> None:
    # Windows' CRT rejects fsync on a descriptor opened read-only.  Reopen in
    # update mode solely for FlushFileBuffers; no payload byte is modified.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> bool:
    """Best-effort directory barrier; Windows lacks portable directory fsync."""

    if os.name == "nt":
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _atomic_bytes(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_file(path)
    return _fsync_directory(path.parent)


def _immutable_bytes(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"conflicting immutable publication: {path}")
        return _fsync_directory(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            # Link publication is fail-if-exists and exposes only the already
            # fsynced complete temporary inode.  It therefore cannot overwrite
            # a conflicting receipt in a duplicate-worker race.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"conflicting immutable publication race: {path}"
                )
        _fsync_file(path)
        return _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ObjectBinding:
    schema_version: str
    role: str
    media_type: str
    path: str
    bytes: int
    sha256: str
    file_fsync: bool
    directory_fsync: bool
    reopened_and_rehashed: bool


class ImmutableObjectStore:
    """One-run immutable object and receipt namespace."""

    def __init__(
        self,
        *,
        repository_root: Path,
        object_root: Path,
        staging_root: Path,
        receipt_root: Path,
        task_id: str,
        run_id: str,
        config_sha256: str,
        plan_sha256: str,
        source_snapshot_sha256: str,
        seed_namespace: str,
        runner_id: str,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.object_root = _inside(
            object_root, self.repository_root, "object_root"
        )
        self.staging_root = _inside(
            staging_root, self.repository_root, "staging_root"
        )
        self.receipt_root = _inside(
            receipt_root, self.repository_root, "receipt_root"
        )
        if len({self.object_root, self.staging_root, self.receipt_root}) != 3:
            raise ValueError("object, staging and receipt roots must be distinct")
        for directory in (self.object_root, self.staging_root, self.receipt_root):
            directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id is required")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        for name, value in (
            ("config_sha256", config_sha256),
            ("plan_sha256", plan_sha256),
            ("source_snapshot_sha256", source_snapshot_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value == "0" * 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a nonzero lowercase SHA-256")
        for name, value in (
            ("seed_namespace", seed_namespace),
            ("runner_id", runner_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or any(
                    character not in
                    "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    for character in value
                )
            ):
                raise ValueError(f"{name} must be one safe non-empty identifier")
        self.task_id = task_id
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.plan_sha256 = plan_sha256
        self.source_snapshot_sha256 = source_snapshot_sha256
        self.seed_namespace = seed_namespace
        self.runner_id = runner_id

    def new_staging_path(self, *, suffix: str = ".bin") -> Path:
        if (
            not isinstance(suffix, str)
            or not suffix.startswith(".")
            or "/" in suffix
            or "\\" in suffix
        ):
            raise ValueError("staging suffix must be one safe extension")
        path = self.staging_root / f"{uuid4().hex}{suffix}"
        # Reserve the name now, so two workers cannot share a staging target.
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        return path

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repository_root).as_posix()

    def adopt_staged_file(
        self,
        staged: Path,
        *,
        role: str,
        media_type: str,
    ) -> ObjectBinding:
        """Publish a same-volume staged file under its content address."""

        staged = _inside(staged, self.staging_root, "staged file")
        _regular_nonsymlink(staged, "staged file")
        if not isinstance(role, str) or not role:
            raise ValueError("object role is required")
        if not isinstance(media_type, str) or "/" not in media_type:
            raise ValueError("object media_type is required")
        _fsync_file(staged)
        size, digest = _sha_file(staged)
        destination = self.object_root / digest[:2] / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        directory_fsync = False
        if destination.exists():
            _regular_nonsymlink(destination, "existing object")
            existing_size, existing_digest = _sha_file(destination)
            if existing_size != size or existing_digest != digest:
                raise RuntimeError("content-address collision or corrupted object")
            staged.unlink()
        else:
            # An exclusive same-volume hard-link publication is safe when
            # several workers discover the same digest concurrently.  It
            # creates no second payload copy; removing the staging name leaves
            # exactly the immutable content-addressed name.  A losing worker
            # revalidates the winner's bytes before removing its own staging
            # name.
            try:
                os.link(staged, destination)
                directory_fsync = _fsync_directory(destination.parent)
                staged.unlink()
            except FileExistsError:
                _regular_nonsymlink(destination, "concurrent object")
                existing_size, existing_digest = _sha_file(destination)
                if existing_size != size or existing_digest != digest:
                    raise RuntimeError(
                        "concurrent content-address collision or corrupted object"
                    )
                staged.unlink()
        _fsync_file(destination)
        reopened_size, reopened_digest = _sha_file(destination)
        if reopened_size != size or reopened_digest != digest:
            raise RuntimeError("published object failed reopen/hash verification")
        return ObjectBinding(
            schema_version=OBJECT_SCHEMA,
            role=role,
            media_type=media_type,
            path=self._relative(destination),
            bytes=size,
            sha256=digest,
            file_fsync=True,
            directory_fsync=directory_fsync,
            reopened_and_rehashed=True,
        )

    def put_bytes(
        self,
        payload: bytes,
        *,
        role: str,
        media_type: str = "application/octet-stream",
    ) -> ObjectBinding:
        """Small-fixture convenience path; production payloads stream to staging."""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        staged = self.new_staging_path()
        with staged.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return self.adopt_staged_file(
            staged,
            role=role,
            media_type=media_type,
        )

    def _object_path(self, binding: Mapping[str, Any]) -> Path:
        digest = binding.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("object binding has invalid sha256")
        expected = self.object_root / digest[:2] / digest
        registered = binding.get("path")
        if not isinstance(registered, str):
            raise ValueError("object binding path missing")
        path = _inside(self.repository_root / registered, self.object_root, "object")
        if path != expected.resolve():
            raise ValueError("object path is not its canonical content address")
        return path

    def verify_object(self, binding: Mapping[str, Any]) -> ObjectBinding:
        required = {
            "schema_version",
            "role",
            "media_type",
            "path",
            "bytes",
            "sha256",
            "file_fsync",
            "directory_fsync",
            "reopened_and_rehashed",
        }
        if set(binding) != required or binding.get("schema_version") != OBJECT_SCHEMA:
            raise ValueError("object binding schema drift")
        path = self._object_path(binding)
        _regular_nonsymlink(path, "object")
        size, digest = _sha_file(path)
        if (
            binding.get("bytes") != size
            or binding.get("sha256") != digest
            or binding.get("file_fsync") is not True
            or binding.get("reopened_and_rehashed") is not True
            or not isinstance(binding.get("directory_fsync"), bool)
        ):
            raise RuntimeError("object binding does not match live bytes")
        return ObjectBinding(**dict(binding))

    def receipt_path(self, chunk_id: str) -> Path:
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or "/" in chunk_id
            or "\\" in chunk_id
            or chunk_id in {".", ".."}
        ):
            raise ValueError("unsafe chunk_id")
        return self.receipt_root / f"{chunk_id}.json"

    def commit_receipt(
        self,
        *,
        cell: Mapping[str, Any],
        objects: Sequence[ObjectBinding],
        expected_rows: int,
        observed_rows: int,
        exception_rows: int,
        missing_rows: int,
        conservation_failures: int,
        source_snapshot_sha256: str,
        runtime_fingerprint: Mapping[str, Any],
        reset_rows: int,
        reset_sidecar_rows: int,
    ) -> dict[str, Any]:
        """Publish one immutable receipt after every object has been rechecked."""

        chunk_id = cell.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise ValueError("cell chunk_id missing")
        expected_rows = int(expected_rows)
        observed_rows = int(observed_rows)
        if expected_rows <= 0 or observed_rows < 0:
            raise ValueError("invalid row denominator")
        diagnostics = {
            "expected_rows": expected_rows,
            "observed_rows": observed_rows,
            "exception_rows": int(exception_rows),
            "missing_rows": int(missing_rows),
            "conservation_failures": int(conservation_failures),
            "reset_rows": int(reset_rows),
            "reset_sidecar_rows": int(reset_sidecar_rows),
        }
        if any(value < 0 for value in diagnostics.values()):
            raise ValueError("negative receipt diagnostic")
        if (
            observed_rows + int(missing_rows) != expected_rows
            or int(exception_rows) > observed_rows
            or int(conservation_failures) > observed_rows
            or int(reset_rows) != int(reset_sidecar_rows)
        ):
            raise ValueError("receipt denominator/sidecar invariant failed")
        roles = [binding.role for binding in objects]
        if len(roles) != len(set(roles)) or not roles:
            raise ValueError("object roles must be non-empty and unique")
        if source_snapshot_sha256 != self.source_snapshot_sha256:
            raise ValueError("receipt source snapshot differs from store lineage")
        try:
            _validate_runtime_fingerprint(
                runtime_fingerprint,
                runner_id=self.runner_id,
                seed_namespace=self.seed_namespace,
            )
        except RuntimeError as exc:
            raise ValueError(
                "receipt runtime fingerprint differs from store lineage"
            ) from exc
        verified = [
            asdict(self.verify_object(asdict(binding))) for binding in objects
        ]
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "plan_sha256": self.plan_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "cell": dict(cell),
            "diagnostics": diagnostics,
            "runtime_fingerprint": dict(runtime_fingerprint),
            "objects": verified,
        }
        receipt["receipt_sha256"] = _sha(receipt)
        payload = _canonical(receipt) + b"\n"
        path = self.receipt_path(chunk_id)
        if path.exists():
            existing = path.read_bytes()
            if existing != payload:
                raise RuntimeError("conflicting immutable receipt replay")
            return self.verify_receipt(path, expected_cell=cell)
        _immutable_bytes(path, payload)
        return self.verify_receipt(path, expected_cell=cell)

    def verify_receipt(
        self,
        path: Path,
        *,
        expected_cell: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = _inside(path, self.receipt_root, "receipt")
        _regular_nonsymlink(path, "receipt")
        receipt = _strict_json_object(path, label="receipt")
        if set(receipt) != RECEIPT_FIELDS:
            raise RuntimeError("receipt top-level schema drift")
        claimed = receipt.get("receipt_sha256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        if claimed != _sha(unsigned):
            raise RuntimeError("receipt self hash mismatch")
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("task_id") != self.task_id
            or receipt.get("run_id") != self.run_id
            or receipt.get("config_sha256") != self.config_sha256
            or receipt.get("plan_sha256") != self.plan_sha256
            or receipt.get("source_snapshot_sha256")
            != self.source_snapshot_sha256
        ):
            raise RuntimeError("receipt lineage mismatch")
        _validate_runtime_fingerprint(
            receipt.get("runtime_fingerprint"),
            runner_id=self.runner_id,
            seed_namespace=self.seed_namespace,
        )
        cell = receipt.get("cell")
        if not isinstance(cell, Mapping) or (
            expected_cell is not None
            and _canonical(cell) != _canonical(expected_cell)
        ):
            raise RuntimeError("receipt cell identity mismatch")
        if path != self.receipt_path(str(cell.get("chunk_id"))).resolve():
            raise RuntimeError("receipt path/chunk mismatch")
        objects = receipt.get("objects")
        if not isinstance(objects, list) or not objects:
            raise RuntimeError("receipt object list missing")
        roles: set[str] = set()
        for binding in objects:
            if not isinstance(binding, Mapping):
                raise RuntimeError("invalid receipt object binding")
            verified = self.verify_object(binding)
            if verified.role in roles:
                raise RuntimeError("duplicate object role in receipt")
            roles.add(verified.role)
        diagnostics = receipt.get("diagnostics")
        required_diagnostics = {
            "expected_rows",
            "observed_rows",
            "exception_rows",
            "missing_rows",
            "conservation_failures",
            "reset_rows",
            "reset_sidecar_rows",
        }
        if (
            not isinstance(diagnostics, Mapping)
            or set(diagnostics) != required_diagnostics
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in diagnostics.values()
            )
        ):
            raise RuntimeError("receipt diagnostics schema drift")
        if (
            diagnostics["observed_rows"] + diagnostics["missing_rows"]
            != diagnostics["expected_rows"]
            or diagnostics["exception_rows"] > diagnostics["observed_rows"]
            or diagnostics["conservation_failures"]
            > diagnostics["observed_rows"]
            or diagnostics["reset_rows"]
            != diagnostics["reset_sidecar_rows"]
        ):
            raise RuntimeError("receipt denominator/sidecar invariant drift")
        if (
            isinstance(cell.get("expected_rows"), bool)
            or not isinstance(cell.get("expected_rows"), int)
            or diagnostics["expected_rows"] != cell["expected_rows"]
        ):
            raise RuntimeError("receipt diagnostic/cell denominator drift")
        return receipt

    def inventory(
        self,
        plan_cells: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Revalidate exactly one receipt per plan cell without copying raw data."""

        expected = {
            str(cell["chunk_id"]): dict(cell)
            for cell in plan_cells
        }
        if len(expected) != len(plan_cells):
            raise RuntimeError("plan contains duplicate chunk IDs")
        discovered = {
            path.stem: path
            for path in self.receipt_root.glob("*.json")
        }
        if set(discovered) != set(expected):
            missing = sorted(set(expected) - set(discovered))
            extra = sorted(set(discovered) - set(expected))
            raise RuntimeError(
                f"receipt coverage mismatch missing={missing[:3]} extra={extra[:3]}"
            )
        receipts: list[dict[str, Any]] = []
        object_digests: set[str] = set()
        totals = {
            "expected_rows": 0,
            "observed_rows": 0,
            "exception_rows": 0,
            "missing_rows": 0,
            "conservation_failures": 0,
            "reset_rows": 0,
            "reset_sidecar_rows": 0,
            "object_bytes_unique": 0,
        }
        digest_bytes: dict[str, int] = {}
        for cell in plan_cells:
            chunk_id = str(cell["chunk_id"])
            receipt = self.verify_receipt(
                discovered[chunk_id],
                expected_cell=cell,
            )
            receipts.append(
                {
                    "chunk_id": chunk_id,
                    "receipt_path": self._relative(discovered[chunk_id]),
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
            for key, value in receipt["diagnostics"].items():
                totals[key] += int(value)
            for binding in receipt["objects"]:
                digest = str(binding["sha256"])
                size = int(binding["bytes"])
                if digest in digest_bytes and digest_bytes[digest] != size:
                    raise RuntimeError("same digest registered with different size")
                digest_bytes[digest] = size
                object_digests.add(digest)
        totals["object_bytes_unique"] = sum(digest_bytes.values())
        incomplete = (
            totals["observed_rows"] != totals["expected_rows"]
            or totals["exception_rows"] != 0
            or totals["missing_rows"] != 0
            or totals["conservation_failures"] != 0
        )
        inventory: dict[str, Any] = {
            "schema_version": INVENTORY_SCHEMA,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "plan_sha256": self.plan_sha256,
            "receipt_count": len(receipts),
            "unique_object_count": len(object_digests),
            "totals": totals,
            "raw_status": (
                "INCOMPLETE_FAIL_CLOSED"
                if incomplete
                else "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
            ),
            "receipts": receipts,
            "monolithic_archive": None,
            "merged_full_csv": None,
            "scientific_verdict": None,
            "qualified_claim": None,
        }
        inventory["inventory_sha256"] = _sha(inventory)
        return inventory


def publish_inventory_and_manifest(
    *,
    repository_root: Path,
    inventory_path: Path,
    manifest_path: Path,
    inventory: Mapping[str, Any],
    claim_fields: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-phase small-metadata finalize; no raw object is copied."""

    repository_root = repository_root.resolve()
    inventory_path = _inside(
        inventory_path, repository_root, "inventory publication"
    )
    manifest_path = _inside(
        manifest_path, repository_root, "manifest publication"
    )
    if inventory_path == manifest_path:
        raise ValueError("inventory and manifest paths must be distinct")
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("inventory schema mismatch")
    claimed = inventory.get("inventory_sha256")
    unsigned_inventory = dict(inventory)
    unsigned_inventory.pop("inventory_sha256", None)
    if claimed != _sha(unsigned_inventory):
        raise RuntimeError("inventory self hash mismatch")
    _immutable_bytes(inventory_path, _canonical(inventory) + b"\n")
    inventory_binding = {
        "path": inventory_path.relative_to(repository_root).as_posix(),
        "bytes": inventory_path.stat().st_size,
        "sha256": _sha_file(inventory_path)[1],
    }
    claims = {str(field): None for field in claim_fields}
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "task_id": inventory["task_id"],
        "run_id": inventory["run_id"],
        "config_sha256": inventory["config_sha256"],
        "plan_sha256": inventory["plan_sha256"],
        "inventory": inventory_binding,
        "raw_status": inventory["raw_status"],
        "scientific_verdict": None,
        "qualified_claim": None,
        "claim_boundary": claims,
        "independent_verifier_required": True,
        "monolithic_archive": None,
        "merged_full_csv": None,
    }
    manifest["manifest_sha256"] = _sha(manifest)
    _immutable_bytes(manifest_path, _canonical(manifest) + b"\n")
    return dict(inventory), manifest


def append_attempt_event(
    path: Path,
    *,
    task_id: str,
    run_id: str,
    event: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one hash-chained and fsynced attempt event."""

    previous = "0" * 64
    sequence = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record.get("sequence") != sequence:
                    raise RuntimeError(
                        f"attempt ledger sequence drift at line {line_number}"
                    )
                claimed = record.get("event_sha256")
                unsigned = dict(record)
                unsigned.pop("event_sha256", None)
                if (
                    record.get("previous_event_sha256") != previous
                    or claimed != _sha(unsigned)
                ):
                    raise RuntimeError(
                        f"attempt ledger hash drift at line {line_number}"
                    )
                previous = str(claimed)
                sequence += 1
    record: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        "task_id": task_id,
        "run_id": run_id,
        "sequence": sequence,
        "previous_event_sha256": previous,
        "event": event,
        "payload": dict(payload),
    }
    record["event_sha256"] = _sha(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


__all__ = [
    "ATTEMPT_SCHEMA",
    "INVENTORY_SCHEMA",
    "ImmutableObjectStore",
    "MANIFEST_SCHEMA",
    "OBJECT_SCHEMA",
    "ObjectBinding",
    "RECEIPT_SCHEMA",
    "append_attempt_event",
    "publish_inventory_and_manifest",
]
