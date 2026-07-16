"""T4.3.2 transactional double bank for complete parametric MAP-LUT images.

Partial transfer bytes live outside both valid bank slots.  A slot becomes a
commit candidate only after transfer CRC/SHA, canonical decoding, image self
verification, version/CAS, timestamp, and selection-hysteresis checks pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import zlib
from dataclasses import dataclass, replace
from typing import Any

from cnn_fpga.runtime.parametric_map_lut import (
    ONLINE_SCOPE,
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
)


SCHEMA_VERSION = "t4.3.2-atomic-parameter-image-bank-v1"
MODEL_SCOPE = "transactional_double_bank_software_contract_not_rtl_or_board"
COMMIT_STATUSES = ("deferred", "committed", "rejected")
_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


class AtomicParameterBankError(RuntimeError):
    """Fail-closed transaction error carrying one stable reason code."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = str(reason)
        super().__init__(self.reason if message is None else f"{self.reason}: {message}")


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_TOKEN.pattern}")
    return value


def _hex_digest(value: object, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value.lower()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def serialize_parameter_image(image: ParametricMAPLUTImage) -> bytes:
    if not isinstance(image, ParametricMAPLUTImage):
        raise TypeError("image must be ParametricMAPLUTImage")
    image.verify()
    return _canonical_bytes(image.to_dict(include_tables=True))


def deserialize_parameter_image(payload: bytes) -> ParametricMAPLUTImage:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicParameterBankError("payload_decode_failed", str(exc)) from exc
    if not isinstance(decoded, dict):
        raise AtomicParameterBankError("payload_schema_mismatch", "top level must be an object")
    expected_keys = {
        "schema_version",
        "online_scope",
        "config",
        "active_bank_version",
        "source_params_sha256",
        "model_mean",
        "model_sigma",
        "table_codes",
        "llr_saturation_count",
        "image_crc32",
        "image_sha256",
    }
    if set(decoded) != expected_keys or decoded.get("online_scope") != ONLINE_SCOPE:
        raise AtomicParameterBankError("payload_schema_mismatch", "unexpected image fields")
    config_payload = decoded.get("config")
    if not isinstance(config_payload, dict):
        raise AtomicParameterBankError("payload_schema_mismatch", "config must be an object")
    try:
        config = ParametricMAPLUTConfig(
            adc_bits=config_payload["adc_bits"],
            address_bits=config_payload["address_bits"],
            llr_integer_bits=config_payload["llr_integer_bits"],
            llr_fractional_bits=config_payload["llr_fractional_bits"],
            lattice=config_payload["lattice"],
            pipeline_latency_cycles=config_payload["pipeline_latency_cycles"],
            initiation_interval_cycles=config_payload["initiation_interval_cycles"],
        )
        if config.to_dict() != config_payload:
            raise ValueError("derived config fields do not match")
        tables = decoded["table_codes"]
        image = ParametricMAPLUTImage(
            config=config,
            active_bank_version=decoded["active_bank_version"],
            source_params_sha256=decoded["source_params_sha256"],
            model_mean=tuple(decoded["model_mean"]),
            model_sigma=tuple(decoded["model_sigma"]),
            table_codes=(tuple(tables[0]), tuple(tables[1])),
            llr_saturation_count=decoded["llr_saturation_count"],
            image_crc32=decoded["image_crc32"],
            image_sha256=decoded["image_sha256"],
            schema_version=decoded["schema_version"],
        )
        image.verify()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AtomicParameterBankError("image_self_verification_failed", str(exc)) from exc
    if serialize_parameter_image(image) != payload:
        raise AtomicParameterBankError("payload_noncanonical")
    return image


@dataclass(frozen=True)
class AtomicParameterBankConfig:
    fast_cycle_ns: int = 5_000
    promotion_good_windows: int = 2
    min_residency_cycles: int = 4_000
    max_payload_age_cycles: int = 8_192
    max_payload_bytes: int = 1 << 20
    safe_boundary_period_cycles: int = 1
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "fast_cycle_ns",
            "promotion_good_windows",
            "min_residency_cycles",
            "max_payload_age_cycles",
            "max_payload_bytes",
            "safe_boundary_period_cycles",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.max_payload_age_cycles < self.min_residency_cycles:
            raise ValueError("max payload age must not be shorter than minimum residency")
        if self.max_payload_age_cycles >= 1 << 16:
            raise ValueError("max payload age must fit the T4.2 16-bit age word")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class ParameterImageManifest:
    transaction_id: str
    selection_key: str
    expected_active_version: int
    new_version: int
    source_window_id: int
    created_epoch: int
    created_timestamp_ns: int
    apply_epoch: int
    payload_length: int
    payload_crc32: str
    payload_sha256: str
    image_crc32: str
    image_sha256: str
    manifest_crc32: str = ""
    manifest_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "transaction_id", _token(self.transaction_id, "transaction_id"))
        object.__setattr__(self, "selection_key", _token(self.selection_key, "selection_key"))
        for name, minimum in (
            ("expected_active_version", 0),
            ("new_version", 1),
            ("source_window_id", 1),
            ("created_epoch", 0),
            ("created_timestamp_ns", 0),
            ("apply_epoch", 1),
            ("payload_length", 1),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.new_version != self.expected_active_version + 1:
            raise ValueError("new_version must equal expected_active_version+1")
        if self.apply_epoch <= self.created_epoch:
            raise ValueError("apply_epoch must be later than created_epoch")
        object.__setattr__(self, "payload_crc32", _hex_digest(self.payload_crc32, "payload_crc32", 8))
        object.__setattr__(self, "payload_sha256", _hex_digest(self.payload_sha256, "payload_sha256", 64))
        object.__setattr__(self, "image_crc32", _hex_digest(self.image_crc32, "image_crc32", 8))
        object.__setattr__(self, "image_sha256", _hex_digest(self.image_sha256, "image_sha256", 64))
        if self.manifest_crc32:
            object.__setattr__(
                self,
                "manifest_crc32",
                _hex_digest(self.manifest_crc32, "manifest_crc32", 8),
            )
        if self.manifest_sha256:
            object.__setattr__(
                self,
                "manifest_sha256",
                _hex_digest(self.manifest_sha256, "manifest_sha256", 64),
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    def unsigned_payload(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("manifest_crc32")
        payload.pop("manifest_sha256")
        return payload

    def verify(self) -> None:
        if not self.manifest_crc32 or not self.manifest_sha256:
            raise AtomicParameterBankError("manifest_integrity_missing")
        payload = _canonical_bytes(self.unsigned_payload())
        expected_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        expected_sha = hashlib.sha256(payload).hexdigest()
        if self.manifest_crc32 != expected_crc:
            raise AtomicParameterBankError("manifest_crc_mismatch")
        if self.manifest_sha256 != expected_sha:
            raise AtomicParameterBankError("manifest_sha256_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def seal_parameter_image_manifest(
    manifest: ParameterImageManifest,
) -> ParameterImageManifest:
    if not isinstance(manifest, ParameterImageManifest):
        raise TypeError("manifest must be ParameterImageManifest")
    payload = _canonical_bytes(manifest.unsigned_payload())
    return replace(
        manifest,
        manifest_crc32=f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_parameter_image_manifest(
    image: ParametricMAPLUTImage,
    *,
    transaction_id: str,
    selection_key: str,
    expected_active_version: int,
    source_window_id: int,
    created_epoch: int,
    apply_epoch: int,
    fast_cycle_ns: int = 5_000,
) -> tuple[ParameterImageManifest, bytes]:
    payload = serialize_parameter_image(image)
    epoch = _integer(created_epoch, "created_epoch")
    cycle_ns = _integer(fast_cycle_ns, "fast_cycle_ns", 1)
    manifest = seal_parameter_image_manifest(ParameterImageManifest(
        transaction_id=transaction_id,
        selection_key=selection_key,
        expected_active_version=expected_active_version,
        new_version=image.active_bank_version,
        source_window_id=source_window_id,
        created_epoch=epoch,
        created_timestamp_ns=epoch * cycle_ns,
        apply_epoch=apply_epoch,
        payload_length=len(payload),
        payload_crc32=f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        image_crc32=image.image_crc32,
        image_sha256=image.image_sha256,
    ))
    return manifest, payload


@dataclass(frozen=True)
class SelectionHysteresisState:
    last_window_id: int = 0
    selection_key: str = ""
    good_run: int = 0
    evidence_window_ids: tuple[int, ...] = ()
    promotable: bool = False


@dataclass(frozen=True)
class StagedParameterImage:
    target_bank: str
    manifest: ParameterImageManifest
    finalized_epoch: int
    image_crc32: str
    image_sha256: str


@dataclass(frozen=True)
class CommitAck:
    ack_sequence: int
    status: str
    accepted: bool
    final: bool
    reason: str
    epoch: int
    requested_version: int
    active_bank: str
    active_version: int
    image_crc32: str
    image_sha256: str

    def __post_init__(self) -> None:
        if self.status not in COMMIT_STATUSES:
            raise ValueError(f"status must be one of {COMMIT_STATUSES}")


@dataclass(frozen=True)
class ActiveImageReadback:
    read_sequence: int
    epoch: int
    active_bank: str
    active_version: int
    activated_epoch: int
    image_crc32: str
    image_sha256: str
    source_window_id: int


@dataclass(frozen=True)
class _BankSlot:
    image: ParametricMAPLUTImage
    payload: bytes
    activated_epoch: int
    source_window_id: int


@dataclass
class _Transfer:
    target_bank: str
    manifest: ParameterImageManifest
    buffer: bytearray
    coverage: bytearray
    written_bytes: int = 0


class AtomicParameterImageBank:
    """Thread-safe two-bank transaction controller with fail-closed activation."""

    def __init__(
        self,
        initial_image: ParametricMAPLUTImage,
        config: AtomicParameterBankConfig | None = None,
    ) -> None:
        self.config = AtomicParameterBankConfig() if config is None else config
        if not isinstance(self.config, AtomicParameterBankConfig):
            raise TypeError("config must be AtomicParameterBankConfig")
        payload = serialize_parameter_image(initial_image)
        self._slots: dict[str, _BankSlot | None] = {
            "A": _BankSlot(initial_image, payload, 0, 0),
            "B": None,
        }
        self._active_bank = "A"
        self._inactive_bank = "B"
        self._epoch = 0
        self._last_activation_epoch = 0
        self._transfer: _Transfer | None = None
        self._pending: StagedParameterImage | None = None
        self._hysteresis = SelectionHysteresisState()
        self._ack_sequence = 0
        self._read_sequence = 0
        self._seen_transaction_ids: set[str] = set()
        self._lock = threading.RLock()

    @property
    def active_bank(self) -> str:
        with self._lock:
            return self._active_bank

    @property
    def inactive_bank(self) -> str:
        with self._lock:
            return self._inactive_bank

    @property
    def active_version(self) -> int:
        with self._lock:
            slot = self._slots[self._active_bank]
            assert slot is not None
            return slot.image.active_bank_version

    @property
    def hysteresis_state(self) -> SelectionHysteresisState:
        with self._lock:
            return self._hysteresis

    @property
    def pending(self) -> StagedParameterImage | None:
        with self._lock:
            return self._pending

    def _advance_epoch(self, epoch: int) -> int:
        actual = _integer(epoch, "epoch")
        if actual < self._epoch:
            raise AtomicParameterBankError("nonmonotonic_epoch")
        self._epoch = actual
        return actual

    def read_active_image(self) -> ParametricMAPLUTImage:
        with self._lock:
            slot = self._slots[self._active_bank]
            assert slot is not None
            return slot.image

    def observe_selection(
        self,
        *,
        window_id: int,
        selection_key: str,
        eligible: bool,
    ) -> SelectionHysteresisState:
        with self._lock:
            window = _integer(window_id, "window_id", 1)
            key = _token(selection_key, "selection_key")
            if not isinstance(eligible, bool):
                raise TypeError("eligible must be boolean")
            previous = self._hysteresis
            if window <= previous.last_window_id:
                raise AtomicParameterBankError("window_sequence_nonmonotonic")
            if not eligible:
                state = SelectionHysteresisState(last_window_id=window)
            elif key == previous.selection_key:
                run = previous.good_run + 1
                evidence = (previous.evidence_window_ids + (window,))[
                    -self.config.promotion_good_windows :
                ]
                state = SelectionHysteresisState(
                    last_window_id=window,
                    selection_key=key,
                    good_run=run,
                    evidence_window_ids=evidence,
                    promotable=run >= self.config.promotion_good_windows,
                )
            else:
                state = SelectionHysteresisState(
                    last_window_id=window,
                    selection_key=key,
                    good_run=1,
                    evidence_window_ids=(window,),
                    promotable=self.config.promotion_good_windows == 1,
                )
            self._hysteresis = state
            return state

    def _validate_manifest(self, manifest: ParameterImageManifest, current_epoch: int) -> None:
        if not isinstance(manifest, ParameterImageManifest):
            raise TypeError("manifest must be ParameterImageManifest")
        manifest.verify()
        if manifest.payload_length > self.config.max_payload_bytes:
            raise AtomicParameterBankError("payload_too_large")
        if manifest.created_timestamp_ns != manifest.created_epoch * self.config.fast_cycle_ns:
            raise AtomicParameterBankError("timestamp_epoch_mismatch")
        if manifest.created_epoch > current_epoch:
            raise AtomicParameterBankError("created_in_future")
        if current_epoch - manifest.created_epoch > self.config.max_payload_age_cycles:
            raise AtomicParameterBankError("payload_stale")
        if manifest.apply_epoch - manifest.created_epoch > self.config.max_payload_age_cycles:
            raise AtomicParameterBankError("apply_epoch_stale")
        if manifest.apply_epoch <= current_epoch:
            raise AtomicParameterBankError("apply_epoch_not_future")
        if manifest.expected_active_version != self.active_version:
            raise AtomicParameterBankError("expected_active_version_mismatch")
        if manifest.new_version != self.active_version + 1:
            raise AtomicParameterBankError("new_version_not_next")
        state = self._hysteresis
        if not state.promotable or state.selection_key != manifest.selection_key:
            raise AtomicParameterBankError("hysteresis_not_satisfied")
        if state.last_window_id != manifest.source_window_id:
            raise AtomicParameterBankError("source_window_not_latest_hysteresis_evidence")

    def begin_stage(self, manifest: ParameterImageManifest, *, current_epoch: int) -> str:
        with self._lock:
            epoch = self._advance_epoch(current_epoch)
            if self._transfer is not None:
                raise AtomicParameterBankError("writer_conflict_transfer_in_progress")
            if self._pending is not None:
                raise AtomicParameterBankError("writer_conflict_pending_commit")
            self._validate_manifest(manifest, epoch)
            if manifest.transaction_id in self._seen_transaction_ids:
                raise AtomicParameterBankError("transaction_replay")
            self._seen_transaction_ids.add(manifest.transaction_id)
            self._transfer = _Transfer(
                target_bank=self._inactive_bank,
                manifest=manifest,
                buffer=bytearray(manifest.payload_length),
                coverage=bytearray(manifest.payload_length),
            )
            return manifest.transaction_id

    def write_chunk(self, transaction_id: str, *, offset: int, chunk: bytes) -> int:
        with self._lock:
            token = _token(transaction_id, "transaction_id")
            transfer = self._transfer
            if transfer is None or transfer.manifest.transaction_id != token:
                raise AtomicParameterBankError("unknown_transaction")
            start = _integer(offset, "offset")
            if not isinstance(chunk, bytes) or len(chunk) == 0:
                raise ValueError("chunk must be non-empty bytes")
            end = start + len(chunk)
            if end > len(transfer.buffer):
                self._transfer = None
                raise AtomicParameterBankError("chunk_out_of_bounds")
            for index, value in enumerate(chunk, start=start):
                if transfer.coverage[index] and transfer.buffer[index] != value:
                    self._transfer = None
                    raise AtomicParameterBankError("conflicting_overlap")
            for index, value in enumerate(chunk, start=start):
                if not transfer.coverage[index]:
                    transfer.written_bytes += 1
                    transfer.coverage[index] = 1
                    transfer.buffer[index] = value
            return transfer.written_bytes

    def finalize_stage(self, transaction_id: str, *, current_epoch: int) -> StagedParameterImage:
        with self._lock:
            epoch = self._advance_epoch(current_epoch)
            token = _token(transaction_id, "transaction_id")
            transfer = self._transfer
            if transfer is None or transfer.manifest.transaction_id != token:
                raise AtomicParameterBankError("unknown_transaction")
            manifest = transfer.manifest
            if epoch - manifest.created_epoch > self.config.max_payload_age_cycles:
                self._transfer = None
                raise AtomicParameterBankError("payload_stale")
            if manifest.expected_active_version != self.active_version:
                self._transfer = None
                raise AtomicParameterBankError("cas_changed_during_transfer")
            if transfer.written_bytes != manifest.payload_length:
                raise AtomicParameterBankError(
                    "payload_incomplete",
                    f"{transfer.written_bytes}/{manifest.payload_length} bytes",
                )
            payload = bytes(transfer.buffer)
            crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
            sha = hashlib.sha256(payload).hexdigest()
            if crc != manifest.payload_crc32:
                self._transfer = None
                raise AtomicParameterBankError("transfer_crc_mismatch")
            if sha != manifest.payload_sha256:
                self._transfer = None
                raise AtomicParameterBankError("transfer_sha256_mismatch")
            try:
                image = deserialize_parameter_image(payload)
            except AtomicParameterBankError:
                self._transfer = None
                raise
            if image.active_bank_version != manifest.new_version:
                self._transfer = None
                raise AtomicParameterBankError("image_version_mismatch")
            if (
                image.image_crc32 != manifest.image_crc32
                or image.image_sha256 != manifest.image_sha256
            ):
                self._transfer = None
                raise AtomicParameterBankError("manifest_image_digest_mismatch")
            slot = _BankSlot(
                image=image,
                payload=payload,
                activated_epoch=-1,
                source_window_id=manifest.source_window_id,
            )
            # This is the only write to a valid bank slot: complete verified image at once.
            self._slots[transfer.target_bank] = slot
            staged = StagedParameterImage(
                target_bank=transfer.target_bank,
                manifest=manifest,
                finalized_epoch=epoch,
                image_crc32=image.image_crc32,
                image_sha256=image.image_sha256,
            )
            self._pending = staged
            self._transfer = None
            return staged

    def _ack(
        self,
        *,
        status: str,
        accepted: bool,
        final: bool,
        reason: str,
        epoch: int,
        requested_version: int,
    ) -> CommitAck:
        self._ack_sequence += 1
        image = self.read_active_image()
        return CommitAck(
            ack_sequence=self._ack_sequence,
            status=status,
            accepted=accepted,
            final=final,
            reason=reason,
            epoch=epoch,
            requested_version=requested_version,
            active_bank=self._active_bank,
            active_version=image.active_bank_version,
            image_crc32=image.image_crc32,
            image_sha256=image.image_sha256,
        )

    def _reject_pending(self, epoch: int, reason: str) -> CommitAck:
        assert self._pending is not None
        requested = self._pending.manifest.new_version
        target = self._pending.target_bank
        self._pending = None
        self._slots[target] = None
        return self._ack(
            status="rejected",
            accepted=False,
            final=True,
            reason=reason,
            epoch=epoch,
            requested_version=requested,
        )

    def commit_if_ready(self, epoch: int, *, safe_boundary: bool) -> CommitAck | None:
        with self._lock:
            actual = self._advance_epoch(epoch)
            if not isinstance(safe_boundary, bool):
                raise TypeError("safe_boundary must be boolean")
            pending = self._pending
            if pending is None or actual < pending.manifest.apply_epoch:
                return None
            requested = pending.manifest.new_version
            if (
                not safe_boundary
                or actual % self.config.safe_boundary_period_cycles != 0
            ):
                return self._ack(
                    status="deferred",
                    accepted=False,
                    final=False,
                    reason="unsafe_cycle_boundary",
                    epoch=actual,
                    requested_version=requested,
                )
            manifest = pending.manifest
            if manifest.expected_active_version != self.active_version:
                return self._reject_pending(actual, "cas_changed_before_commit")
            if actual - manifest.created_epoch > self.config.max_payload_age_cycles:
                return self._reject_pending(actual, "payload_stale_before_commit")
            state = self._hysteresis
            if not state.promotable or state.selection_key != manifest.selection_key:
                return self._reject_pending(actual, "hysteresis_invalidated")
            if actual - self._last_activation_epoch < self.config.min_residency_cycles:
                return self._ack(
                    status="deferred",
                    accepted=False,
                    final=False,
                    reason="minimum_residency_not_met",
                    epoch=actual,
                    requested_version=requested,
                )
            slot = self._slots[pending.target_bank]
            if slot is None or slot.image.active_bank_version != requested:
                return self._reject_pending(actual, "staged_slot_invalid")
            old_active = self._active_bank
            new_active = pending.target_bank
            self._slots[new_active] = replace(slot, activated_epoch=actual)
            self._active_bank = new_active
            self._inactive_bank = old_active
            self._last_activation_epoch = actual
            self._pending = None
            last_window = state.last_window_id
            self._hysteresis = SelectionHysteresisState(last_window_id=last_window)
            return self._ack(
                status="committed",
                accepted=True,
                final=True,
                reason="commit_applied",
                epoch=actual,
                requested_version=requested,
            )

    def readback(self, *, epoch: int | None = None) -> ActiveImageReadback:
        with self._lock:
            actual = self._epoch if epoch is None else self._advance_epoch(epoch)
            slot = self._slots[self._active_bank]
            assert slot is not None
            self._read_sequence += 1
            return ActiveImageReadback(
                read_sequence=self._read_sequence,
                epoch=actual,
                active_bank=self._active_bank,
                active_version=slot.image.active_bank_version,
                activated_epoch=slot.activated_epoch,
                image_crc32=slot.image.image_crc32,
                image_sha256=slot.image.image_sha256,
                source_window_id=slot.source_window_id,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "epoch": self._epoch,
                "active_bank": self._active_bank,
                "inactive_bank": self._inactive_bank,
                "active_version": self.active_version,
                "slot_versions": {
                    name: None if slot is None else slot.image.active_bank_version
                    for name, slot in self._slots.items()
                },
                "transfer": None
                if self._transfer is None
                else {
                    "transaction_id": self._transfer.manifest.transaction_id,
                    "target_bank": self._transfer.target_bank,
                    "written_bytes": self._transfer.written_bytes,
                    "payload_length": self._transfer.manifest.payload_length,
                },
                "pending_version": None
                if self._pending is None
                else self._pending.manifest.new_version,
                "hysteresis": dict(self._hysteresis.__dict__),
                "model_scope": self.config.model_scope,
            }


def verify_commit_ack_readback(ack: CommitAck, readback: ActiveImageReadback) -> bool:
    if not isinstance(ack, CommitAck) or not isinstance(readback, ActiveImageReadback):
        raise TypeError("ack/readback types are required")
    if not ack.accepted or ack.status != "committed" or not ack.final:
        raise AtomicParameterBankError("ack_not_committed")
    fields_match = (
        ack.active_bank == readback.active_bank
        and ack.active_version == readback.active_version
        and ack.epoch == readback.activated_epoch
        and ack.image_crc32 == readback.image_crc32
        and ack.image_sha256 == readback.image_sha256
    )
    if not fields_match:
        raise AtomicParameterBankError("readback_mismatch")
    return True


__all__ = [
    "SCHEMA_VERSION",
    "MODEL_SCOPE",
    "AtomicParameterBankError",
    "AtomicParameterBankConfig",
    "ParameterImageManifest",
    "SelectionHysteresisState",
    "StagedParameterImage",
    "CommitAck",
    "ActiveImageReadback",
    "AtomicParameterImageBank",
    "serialize_parameter_image",
    "deserialize_parameter_image",
    "build_parameter_image_manifest",
    "seal_parameter_image_manifest",
    "verify_commit_ack_readback",
]
