from __future__ import annotations

import struct

import pytest

import cnn_fpga.hwio.route_a_uart_protocol as protocol


def _response() -> bytes:
    frame = bytearray(protocol.RESPONSE_BYTES)
    frame[:2] = protocol.RESPONSE_MAGIC
    frame[2] = protocol.PROTOCOL_VERSION
    frame[3] = protocol.CMD_EXECUTE
    frame[4:8] = (9).to_bytes(4, "little")
    frame[8] = protocol.RESPONSE_PAYLOAD_BYTES
    frame[12] = protocol.STATUS_OK
    frame[13] = protocol.CMD_EXECUTE
    frame[14] = protocol.PROTOCOL_VERSION
    frame[16:24] = (100).to_bytes(8, "little")
    frame[24:32] = (107).to_bytes(8, "little")
    frame[47:57] = (1).to_bytes(10, "little")
    frame[86:88] = (2).to_bytes(2, "little")
    frame[88:90] = (3).to_bytes(2, "little")
    frame[90:92] = (4).to_bytes(2, "little")
    frame[-4:] = struct.pack("<I", protocol.crc32_ieee(frame[:-4]))
    return bytes(frame)


def test_request_roundtrip_covers_high_payload_bits() -> None:
    inputs = protocol.RouteAInputs(
        in_word=(1 << 57) | 0x155,
        host_commit_valid=True,
        host_commit_version=0xA55A,
        cfg_we=True,
        cfg_address=0x1FF,
        cfg_data=(1 << 22) - 1,
        posterior_valid=True,
        p_normal=17,
        p_smooth=23,
        p_calibration=31,
        p_burst=37,
        policy_ood_code=0xC0,
        router_boundary=True,
        leakage_event=True,
        lkg_bank=True,
    )
    frame = protocol.encode_request(0xDEADBEEF, protocol.CMD_EXECUTE, inputs)
    decoded = protocol.decode_request(frame)
    assert len(frame) == 40
    assert decoded.sequence == 0xDEADBEEF
    assert decoded.inputs == inputs


def test_request_rejects_crc_and_reserved_payload_bits() -> None:
    bad_crc = bytearray(protocol.encode_request(0, protocol.CMD_STATUS))
    bad_crc[-1] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        protocol.decode_request(bytes(bad_crc))

    reserved = bytearray(protocol.encode_request(0, protocol.CMD_STATUS))
    reserved[35] = 0x80
    reserved[-4:] = struct.pack("<I", protocol.crc32_ieee(reserved[:-4]))
    with pytest.raises(ValueError, match="reserved bits"):
        protocol.decode_request(bytes(reserved))


def test_compact_response_decodes_latency_action_and_counters() -> None:
    decoded = protocol.decode_response(_response())
    assert len(_response()) == 96
    assert decoded.sequence == 9
    assert decoded.action_latency_cycles == 7
    assert decoded.route_action_word == 1
    assert (
        decoded.rx_crc_errors,
        decoded.sequence_errors,
        decoded.uart_framing_errors,
    ) == (2, 3, 4)


def test_response_rejects_crc_corruption() -> None:
    frame = bytearray(_response())
    frame[47] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        protocol.decode_response(bytes(frame))

