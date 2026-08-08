#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "route_a_uart_rx_model.cc"

using Top = cxxrtl_design::p_route__a__uart__rx;
static constexpr unsigned CLKS_PER_BIT = 9;

static void tick(Top &top, std::vector<uint8_t> &received, unsigned &errors) {
    top.p_clk.set<uint32_t>(0);
    top.step();
    top.p_clk.set<uint32_t>(1);
    top.step();
    if (top.p_byte__valid.get<uint32_t>())
        received.push_back(static_cast<uint8_t>(top.p_byte__data.get<uint32_t>()));
    errors += top.p_framing__error.get<uint32_t>();
}

static void drive(Top &top, std::vector<uint8_t> &received, unsigned &errors,
                  bool level, unsigned clocks) {
    top.p_rx__pin.set<uint32_t>(level ? 1 : 0);
    for (unsigned index = 0; index < clocks; ++index)
        tick(top, received, errors);
}

static void send(Top &top, std::vector<uint8_t> &received, unsigned &errors,
                 uint8_t value, bool stop = true) {
    drive(top, received, errors, false, CLKS_PER_BIT);
    for (unsigned bit = 0; bit < 8; ++bit)
        drive(top, received, errors, (value >> bit) & 1U, CLKS_PER_BIT);
    drive(top, received, errors, stop, CLKS_PER_BIT);
    drive(top, received, errors, true, 2);
}

int main() {
    Top top;
    std::vector<uint8_t> received;
    unsigned errors = 0;
    top.p_rx__pin.set<uint32_t>(1);
    top.p_reset__n.set<uint32_t>(0);
    for (unsigned index = 0; index < 4; ++index) tick(top, received, errors);
    top.p_reset__n.set<uint32_t>(1);
    for (unsigned index = 0; index < 4; ++index) tick(top, received, errors);
    for (uint8_t value : {uint8_t{0xa5}, uint8_t{0x5a}, uint8_t{0x00}, uint8_t{0xff}, uint8_t{0x81}})
        send(top, received, errors, value);
    send(top, received, errors, 0x33, false);
    drive(top, received, errors, true, 40);
    if (received != std::vector<uint8_t>({0xa5, 0x5a, 0x00, 0xff, 0x81}))
        throw std::runtime_error("UART RX byte-level waveform mismatch");
    if (errors != 1)
        throw std::runtime_error("UART RX framing-error pulse mismatch");
    std::cout << "received=5,framing_errors=1\n";
    return 0;
}
