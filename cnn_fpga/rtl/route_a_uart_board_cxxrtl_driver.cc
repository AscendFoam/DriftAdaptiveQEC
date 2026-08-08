#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef ROUTE_A_UART_MODEL_HEADER
#define ROUTE_A_UART_MODEL_HEADER "route_a_uart_board_model.cc"
#endif
#include ROUTE_A_UART_MODEL_HEADER

using Top = cxxrtl_design::p_route__a__uart__board__top;

#ifndef UART_CLKS_PER_BIT
#define UART_CLKS_PER_BIT 9
#endif
static constexpr unsigned CLKS_PER_BIT = UART_CLKS_PER_BIT;
static constexpr size_t REQUEST_BYTES = 40;
static constexpr size_t RESPONSE_BYTES = 96;

static uint32_t crc32_byte(uint32_t crc, uint8_t data) {
    crc ^= data;
    for (unsigned bit = 0; bit < 8; ++bit)
        crc = (crc & 1U) ? ((crc >> 1) ^ 0xedb88320U) : (crc >> 1);
    return crc;
}

static uint32_t crc32(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffffU;
    for (size_t index = 0; index < size; ++index)
        crc = crc32_byte(crc, data[index]);
    return ~crc;
}

static uint16_t crc16(const uint8_t *data, size_t size) {
    uint16_t crc = 0xffffU;
    for (size_t index = 0; index < size; ++index) {
        crc ^= static_cast<uint16_t>(data[index]) << 8;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc & 0x8000U) ? static_cast<uint16_t>((crc << 1) ^ 0x1021U)
                                  : static_cast<uint16_t>(crc << 1);
    }
    return crc;
}

static void put_le(std::array<uint8_t, REQUEST_BYTES> &frame, size_t offset,
                   uint64_t value, size_t bytes) {
    for (size_t index = 0; index < bytes; ++index)
        frame[offset + index] = static_cast<uint8_t>(value >> (8 * index));
}

static uint64_t get_le(const std::vector<uint8_t> &frame, size_t offset, size_t bytes) {
    uint64_t value = 0;
    for (size_t index = 0; index < bytes; ++index)
        value |= static_cast<uint64_t>(frame.at(offset + index)) << (8 * index);
    return value;
}

static std::array<uint8_t, REQUEST_BYTES> request(uint32_t sequence, uint8_t command,
                                                  uint64_t input_word = 0) {
    std::array<uint8_t, REQUEST_BYTES> frame{};
    frame[0] = 0xa5;
    frame[1] = 0x5a;
    frame[2] = 1;
    frame[3] = command;
    put_le(frame, 4, sequence, 4);
    frame[8] = 24;
    unsigned __int128 payload = input_word;
    payload |= static_cast<unsigned __int128>(1) << 58;  // safe boundary
    payload |= static_cast<unsigned __int128>(1) << 111; // bank0 trusted
    payload |= static_cast<unsigned __int128>(1) << 112; // bank1 trusted
    payload |= static_cast<unsigned __int128>(1) << 113; // posterior valid
    payload |= static_cast<unsigned __int128>(255) << 114; // normal posterior
    for (size_t index = 0; index < 16; ++index)
        frame[12 + index] = static_cast<uint8_t>(payload >> (8 * index));
    // Bits 128..161 are zero in this nominal request.
    const uint32_t crc = crc32(frame.data(), REQUEST_BYTES - 4);
    put_le(frame, REQUEST_BYTES - 4, crc, 4);
    return frame;
}

struct SerialMonitor {
    enum State { IDLE, START, DATA, STOP } state = IDLE;
    unsigned count = 0;
    unsigned bit = 0;
    uint8_t shift = 0;
    std::vector<uint8_t> bytes;

    void sample(bool line) {
        switch (state) {
        case IDLE:
            if (!line) {
                state = START;
                count = 0;
            }
            break;
        case START:
            if (++count == (CLKS_PER_BIT + 1) / 2) {
                if (line) state = IDLE;
                else { state = DATA; count = 0; bit = 0; shift = 0; }
            }
            break;
        case DATA:
            if (++count == CLKS_PER_BIT) {
                if (line) shift |= static_cast<uint8_t>(1U << bit);
                count = 0;
                if (++bit == 8) state = STOP;
            }
            break;
        case STOP:
            if (++count == CLKS_PER_BIT) {
                if (!line) throw std::runtime_error("board TX emitted a framing error");
                bytes.push_back(shift);
                state = IDLE;
                count = 0;
            }
            break;
        }
    }
};

static void tick(Top &top, SerialMonitor &monitor) {
    top.p_clk.set<uint32_t>(0);
    top.step();
    top.p_clk.set<uint32_t>(1);
    top.step();
    monitor.sample(top.p_uart__tx.get<uint32_t>() != 0);
}

static void drive_level(Top &top, SerialMonitor &monitor, bool level, unsigned clocks) {
    top.p_uart__rx.set<uint32_t>(level ? 1 : 0);
    for (unsigned index = 0; index < clocks; ++index)
        tick(top, monitor);
}

static void send_byte(Top &top, SerialMonitor &monitor, uint8_t value, bool good_stop = true) {
    drive_level(top, monitor, false, CLKS_PER_BIT);
    for (unsigned bit = 0; bit < 8; ++bit)
        drive_level(top, monitor, (value >> bit) & 1U, CLKS_PER_BIT);
    drive_level(top, monitor, good_stop, CLKS_PER_BIT);
    drive_level(top, monitor, true, 2);
}

static void send_frame(Top &top, SerialMonitor &monitor,
                       const std::array<uint8_t, REQUEST_BYTES> &frame) {
    for (uint8_t byte : frame)
        send_byte(top, monitor, byte);
}

static std::vector<uint8_t> wait_response(Top &top, SerialMonitor &monitor,
                                          size_t previous_count) {
    const uint64_t limit = 30000;
    for (uint64_t cycle = 0; cycle < limit; ++cycle) {
        if (monitor.bytes.size() >= previous_count + RESPONSE_BYTES)
            break;
        tick(top, monitor);
    }
    if (monitor.bytes.size() < previous_count + RESPONSE_BYTES)
#ifdef UART_BLACKBOX_INTERNALS
        throw std::runtime_error(
            "timed out waiting for response: bytes=" +
            std::to_string(monitor.bytes.size() - previous_count));
#else
        throw std::runtime_error(
            "timed out waiting for response: bytes=" + std::to_string(monitor.bytes.size() - previous_count) +
            " control_state=" + std::to_string(top.p_control__state.get<uint32_t>()) +
            " rx_index=" + std::to_string(top.p_rx__index.get<uint32_t>()) +
            " expected_seq=" + std::to_string(top.p_expected__sequence.get<uint32_t>()) +
            " crc_index=" + std::to_string(top.p_response__crc__index.get<uint32_t>()) +
            " tx_valid=" + std::to_string(top.p_tx__byte__valid.get<uint32_t>())
        );
#endif
    return std::vector<uint8_t>(monitor.bytes.begin() + previous_count,
                                monitor.bytes.begin() + previous_count + RESPONSE_BYTES);
}

static void verify_response_crc(const std::vector<uint8_t> &response) {
    if (response.size() != RESPONSE_BYTES || response[0] != 0x5a ||
        response[1] != 0xa5 || response[2] != 1 || response[8] != 80)
        throw std::runtime_error("invalid response framing/header");
    const uint32_t expected = crc32(response.data(), RESPONSE_BYTES - 4);
    if (get_le(response, RESPONSE_BYTES - 4, 4) != expected)
        throw std::runtime_error("response CRC mismatch");
}

int main(int argc, char **) {
    Top top;
    SerialMonitor monitor;
    top.p_uart__rx.set<uint32_t>(1);
    top.p_reset__n.set<uint32_t>(0);
    for (unsigned index = 0; index < 4; ++index) tick(top, monitor);
    top.p_reset__n.set<uint32_t>(1);
    for (unsigned index = 0; index < 4; ++index) tick(top, monitor);

    uint64_t payload = 0x155U;
    payload |= static_cast<uint64_t>(1) << 40;
    payload |= static_cast<uint64_t>(1) << 41;
    uint8_t payload_bytes[6]{};
    for (unsigned index = 0; index < 6; ++index)
        payload_bytes[index] = static_cast<uint8_t>(payload >> (8 * index));
    const uint64_t input_word = payload | (static_cast<uint64_t>(crc16(payload_bytes, 6)) << 42);

    size_t cursor = monitor.bytes.size();
    const auto execute0 = request(0, 1, input_word);
    send_frame(top, monitor, execute0);
    if (argc > 1) {
#ifdef UART_BLACKBOX_INTERNALS
        std::cout << "after_request,captured_tx_bytes=" << monitor.bytes.size() << "\n";
#else
        for (unsigned index = 0; index < 20; ++index)
            tick(top, monitor);
        std::cout << "after_request,control_state=" << top.p_control__state.get<uint32_t>()
                  << ",rx_index=" << top.p_rx__index.get<uint32_t>()
                  << ",expected_seq=" << top.p_expected__sequence.get<uint32_t>()
                  << ",crc_index=" << top.p_response__crc__index.get<uint32_t>()
                  << ",tx_valid=" << top.p_tx__byte__valid.get<uint32_t>()
                  << ",action_valid=" << (top.p_route__action__word.curr.data[0] & 1U)
                  << ",captured_tx_bytes=" << monitor.bytes.size() << "\n";
#endif
        return 0;
    }
    auto response0 = wait_response(top, monitor, cursor);
    verify_response_crc(response0);
    if (response0[12] != 0 || response0[13] != 1 || get_le(response0, 4, 4) != 0)
        throw std::runtime_error("nominal execute response is not STATUS_OK");
    const uint64_t latency = get_le(response0, 24, 8) - get_le(response0, 16, 8);
    if (latency != 7)
        throw std::runtime_error("unexpected source-to-action core-cycle count");
    if ((response0[47] & 1U) == 0)
        throw std::runtime_error("route action word is not valid");

    // An exact duplicate is acknowledged without re-execution.  The compact
    // response deliberately avoids a whole-frame replay register.
    cursor = monitor.bytes.size();
    send_frame(top, monitor, execute0);
    auto duplicate = wait_response(top, monitor, cursor);
    verify_response_crc(duplicate);
    if (duplicate[12] != 0x01 || (duplicate[47] & 1U) != 0)
        throw std::runtime_error("duplicate request was not idempotently rejected");

    // Bad CRC must not consume sequence 1.
    auto bad_crc = request(1, 2);
    bad_crc[39] ^= 0x80;
    cursor = monitor.bytes.size();
    send_frame(top, monitor, bad_crc);
    auto crc_response = wait_response(top, monitor, cursor);
    verify_response_crc(crc_response);
    if (crc_response[12] != 0x10 || get_le(crc_response, 86, 2) != 1)
        throw std::runtime_error("bad CRC was not counted/rejected");

    // A gap sequence is rejected and likewise does not consume sequence 1.
    cursor = monitor.bytes.size();
    send_frame(top, monitor, request(3, 2));
    auto seq_response = wait_response(top, monitor, cursor);
    verify_response_crc(seq_response);
    if (seq_response[12] != 0x14 || get_le(seq_response, 88, 2) != 1)
        throw std::runtime_error("bad sequence was not counted/rejected");

    cursor = monitor.bytes.size();
    send_frame(top, monitor, request(1, 2));
    auto status1 = wait_response(top, monitor, cursor);
    verify_response_crc(status1);
    if (status1[12] != 0)
        throw std::runtime_error("valid sequence after negative frames did not recover");

    // A physical framing fault resets the byte parser.  The next valid frame
    // proves resynchronization and exposes the saturated-safe counter.
    send_byte(top, monitor, 0xa5, false);
    drive_level(top, monitor, true, 40);
    cursor = monitor.bytes.size();
    send_frame(top, monitor, request(2, 2));
    auto status2 = wait_response(top, monitor, cursor);
    verify_response_crc(status2);
    if (status2[12] != 0 || get_le(status2, 90, 2) != 1)
        throw std::runtime_error("UART framing recovery/counter failed");

    std::cout << "responses=6,execute_latency_cycles=" << latency
              << ",duplicate_idempotent=1,crc_errors=" << get_le(status2, 86, 2)
              << ",sequence_errors=" << get_le(status2, 88, 2)
              << ",framing_errors=" << get_le(status2, 90, 2)
              << ",response_crc_errors=0\n";
    return 0;
}
