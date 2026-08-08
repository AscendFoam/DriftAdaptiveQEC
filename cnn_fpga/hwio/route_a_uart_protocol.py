"""Fixed-size Route-A UART replay protocol for the Tang Nano 20K candidate top.

This module is a host-side codec and evidence boundary.  It does not claim a
physical link is present: board qualification requires the same frames to be
observed through a real BL616/UART path and tied to a bitstream manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib


PROTOCOL_VERSION = 1
REQUEST_MAGIC = b"\xA5\x5A"
RESPONSE_MAGIC = b"\x5A\xA5"
REQUEST_BYTES = 40
REQUEST_PAYLOAD_BYTES = 24
RESPONSE_BYTES = 96
RESPONSE_PAYLOAD_BYTES = 80

CMD_EXECUTE = 0x01
CMD_STATUS = 0x02
CMD_CLEAR_COUNTERS = 0x03
VALID_COMMANDS = frozenset((CMD_EXECUTE, CMD_STATUS, CMD_CLEAR_COUNTERS))

STATUS_OK = 0x00
STATUS_DUPLICATE_REPLAY = 0x01
STATUS_BAD_CRC = 0x10
STATUS_BAD_VERSION = 0x11
STATUS_BAD_LENGTH = 0x12
STATUS_BAD_RESERVED = 0x13
STATUS_BAD_SEQUENCE = 0x14
STATUS_BAD_COMMAND = 0x15
STATUS_RX_TIMEOUT = 0x16
STATUS_UART_FRAMING = 0x17


def crc32_ieee(data: bytes) -> int:
    """Return the standard reflected CRC-32/IEEE value used on the wire."""

    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass(frozen=True)
class RouteAInputs:
    in_word: int = 0
    safe_boundary: bool = True
    host_commit_valid: bool = False
    host_commit_bank: bool = False
    host_commit_version: int = 0
    cfg_we: bool = False
    cfg_bank: bool = False
    cfg_phase: bool = False
    cfg_address: int = 0
    cfg_data: int = 0
    bank0_trusted: bool = True
    bank1_trusted: bool = True
    posterior_valid: bool = False
    p_normal: int = 255
    p_smooth: int = 0
    p_calibration: int = 0
    p_burst: int = 0
    policy_ood_code: int = 0
    router_boundary: bool = False
    window_prequential_win: bool = False
    integrity_fault: bool = False
    version_fault: bool = False
    integrity_clear: bool = False
    leakage_event: bool = False
    reset_ack: bool = False
    lkg_bank: bool = False

    def __post_init__(self) -> None:
        ranges = {
            "in_word": (self.in_word, 58),
            "host_commit_version": (self.host_commit_version, 16),
            "cfg_address": (self.cfg_address, 9),
            "cfg_data": (self.cfg_data, 22),
            "p_normal": (self.p_normal, 8),
            "p_smooth": (self.p_smooth, 8),
            "p_calibration": (self.p_calibration, 8),
            "p_burst": (self.p_burst, 8),
            "policy_ood_code": (self.policy_ood_code, 8),
        }
        for name, (value, bits) in ranges.items():
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << bits):
                raise ValueError(f"{name} must be an unsigned {bits}-bit integer")

    def packed_value(self) -> int:
        value = self.in_word
        fields = (
            (58, self.safe_boundary),
            (59, self.host_commit_valid),
            (60, self.host_commit_bank),
            (61, self.host_commit_version, 16),
            (77, self.cfg_we),
            (78, self.cfg_bank),
            (79, self.cfg_phase),
            (80, self.cfg_address, 9),
            (89, self.cfg_data, 22),
            (111, self.bank0_trusted),
            (112, self.bank1_trusted),
            (113, self.posterior_valid),
            (114, self.p_normal, 8),
            (122, self.p_smooth, 8),
            (130, self.p_calibration, 8),
            (138, self.p_burst, 8),
            (146, self.policy_ood_code, 8),
            (154, self.router_boundary),
            (155, self.window_prequential_win),
            (156, self.integrity_fault),
            (157, self.version_fault),
            (158, self.integrity_clear),
            (159, self.leakage_event),
            (160, self.reset_ack),
            (161, self.lkg_bank),
        )
        for field in fields:
            offset, field_value, *width = field
            value |= int(field_value) << offset
            if width and int(field_value) >= (1 << width[0]):
                raise AssertionError("field range escaped validation")
        return value

    def to_bytes(self) -> bytes:
        return self.packed_value().to_bytes(REQUEST_PAYLOAD_BYTES, "little")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "RouteAInputs":
        if len(payload) != REQUEST_PAYLOAD_BYTES:
            raise ValueError("Route-A payload must be exactly 24 bytes")
        value = int.from_bytes(payload, "little")
        if value >> 162:
            raise ValueError("Route-A payload reserved bits 162..191 must be zero")

        def bit(offset: int) -> bool:
            return bool((value >> offset) & 1)

        def bits(offset: int, width: int) -> int:
            return (value >> offset) & ((1 << width) - 1)

        return cls(
            in_word=bits(0, 58), safe_boundary=bit(58),
            host_commit_valid=bit(59), host_commit_bank=bit(60),
            host_commit_version=bits(61, 16), cfg_we=bit(77),
            cfg_bank=bit(78), cfg_phase=bit(79), cfg_address=bits(80, 9),
            cfg_data=bits(89, 22), bank0_trusted=bit(111),
            bank1_trusted=bit(112), posterior_valid=bit(113),
            p_normal=bits(114, 8), p_smooth=bits(122, 8),
            p_calibration=bits(130, 8), p_burst=bits(138, 8),
            policy_ood_code=bits(146, 8), router_boundary=bit(154),
            window_prequential_win=bit(155), integrity_fault=bit(156),
            version_fault=bit(157), integrity_clear=bit(158),
            leakage_event=bit(159), reset_ack=bit(160), lkg_bank=bit(161),
        )


@dataclass(frozen=True)
class RequestFrame:
    sequence: int
    command: int
    inputs: RouteAInputs


def encode_request(sequence: int, command: int, inputs: RouteAInputs | None = None) -> bytes:
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("sequence must be uint32")
    if command not in VALID_COMMANDS:
        raise ValueError("unknown Route-A UART command")
    payload = (inputs or RouteAInputs()).to_bytes()
    header = REQUEST_MAGIC + bytes((PROTOCOL_VERSION, command)) + struct.pack("<I", sequence)
    body = header + bytes((REQUEST_PAYLOAD_BYTES, 0, 0, 0)) + payload
    assert len(body) == REQUEST_BYTES - 4
    return body + struct.pack("<I", crc32_ieee(body))


def decode_request(frame: bytes) -> RequestFrame:
    if len(frame) != REQUEST_BYTES:
        raise ValueError("request frame length mismatch")
    if frame[:2] != REQUEST_MAGIC:
        raise ValueError("request magic mismatch")
    if frame[2] != PROTOCOL_VERSION:
        raise ValueError("request protocol version mismatch")
    if frame[3] not in VALID_COMMANDS:
        raise ValueError("request command mismatch")
    if frame[8] != REQUEST_PAYLOAD_BYTES:
        raise ValueError("request payload length mismatch")
    if any(frame[9:12]):
        raise ValueError("request reserved header bytes must be zero")
    if int.from_bytes(frame[-4:], "little") != crc32_ieee(frame[:-4]):
        raise ValueError("request CRC mismatch")
    return RequestFrame(
        sequence=int.from_bytes(frame[4:8], "little"),
        command=frame[3],
        inputs=RouteAInputs.from_bytes(frame[12:36]),
    )


@dataclass(frozen=True)
class BoardResponse:
    sequence: int
    status: int
    command_echo: int
    flags: int
    start_cycle: int
    action_cycle: int
    out_word: int
    route_action_word: int
    route_state_word: int
    route_version_word: int
    map_llr: int
    map_address: int
    active_version: int
    debug_flags: int
    rx_crc_errors: int
    sequence_errors: int
    uart_framing_errors: int

    @property
    def action_latency_cycles(self) -> int:
        return (self.action_cycle - self.start_cycle) & 0xFFFFFFFFFFFFFFFF


def decode_response(frame: bytes) -> BoardResponse:
    if len(frame) != RESPONSE_BYTES:
        raise ValueError("response frame length mismatch")
    if frame[:2] != RESPONSE_MAGIC:
        raise ValueError("response magic mismatch")
    if frame[2] != PROTOCOL_VERSION:
        raise ValueError("response protocol version mismatch")
    if frame[8] != RESPONSE_PAYLOAD_BYTES or any(frame[9:12]):
        raise ValueError("response header length/reserved mismatch")
    if int.from_bytes(frame[-4:], "little") != crc32_ieee(frame[:-4]):
        raise ValueError("response CRC mismatch")
    payload = frame[12:92]
    return BoardResponse(
        sequence=int.from_bytes(frame[4:8], "little"),
        status=payload[0], command_echo=payload[1], flags=payload[3],
        start_cycle=int.from_bytes(payload[4:12], "little"),
        action_cycle=int.from_bytes(payload[12:20], "little"),
        out_word=int.from_bytes(payload[20:35], "little"),
        route_action_word=int.from_bytes(payload[35:45], "little"),
        route_state_word=int.from_bytes(payload[45:57], "little"),
        route_version_word=int.from_bytes(payload[57:65], "little"),
        map_llr=int.from_bytes(payload[65:68], "little") & ((1 << 22) - 1),
        map_address=int.from_bytes(payload[68:70], "little") & 0x1FF,
        active_version=int.from_bytes(payload[70:72], "little"),
        debug_flags=payload[72],
        rx_crc_errors=int.from_bytes(payload[74:76], "little"),
        sequence_errors=int.from_bytes(payload[76:78], "little"),
        uart_framing_errors=int.from_bytes(payload[78:80], "little"),
    )


__all__ = [
    "BoardResponse", "CMD_CLEAR_COUNTERS", "CMD_EXECUTE", "CMD_STATUS",
    "PROTOCOL_VERSION", "REQUEST_BYTES", "REQUEST_PAYLOAD_BYTES",
    "RESPONSE_BYTES", "RESPONSE_PAYLOAD_BYTES", "RequestFrame", "RouteAInputs",
    "STATUS_BAD_COMMAND", "STATUS_BAD_CRC", "STATUS_BAD_LENGTH",
    "STATUS_BAD_RESERVED", "STATUS_BAD_SEQUENCE", "STATUS_BAD_VERSION",
    "STATUS_DUPLICATE_REPLAY", "STATUS_OK", "crc32_ieee", "decode_request",
    "decode_response", "encode_request",
]
