#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include "student_model.cc"

template <size_t Bits>
static std::string hex_wire(const cxxrtl::wire<Bits> &wire) {
    const std::string encoded = wire.curr.str();
    const auto separator = encoded.find('\'');
    if (separator == std::string::npos)
        throw std::runtime_error("unexpected CXXRTL value encoding");
    return encoded.substr(separator + 1);
}

static void settle_low(cxxrtl_design::p_low__dimensional__student__kernel &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(cxxrtl_design::p_low__dimensional__student__kernel &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

static int32_t signed18(uint32_t value) {
    value &= ((uint32_t{1} << 18) - 1);
    return (value & (uint32_t{1} << 17)) ?
        static_cast<int32_t>(value) - (int32_t{1} << 18) :
        static_cast<int32_t>(value);
}

int main() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    cxxrtl_design::p_low__dimensional__student__kernel top;
    top.p_start.set<uint32_t>(0);
    top.p_outcome__e.set<uint32_t>(0);
    top.p_health__ok.set<uint32_t>(1);
    top.p_reset__n.set<uint32_t>(0);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<uint32_t>(1);

    std::cout << "step,outcome_e,health_ok,latency_cycles,state_hex";
    for (int index = 0; index < 15; ++index)
        std::cout << ",out" << index;
    std::cout << "\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        uint32_t step = 0, outcome = 0, health = 0;
        if (!(fields >> step >> outcome >> health))
            throw std::runtime_error("invalid input row: " + line);
        top.p_outcome__e.set<uint32_t>(outcome);
        top.p_health__ok.set<uint32_t>(health);
        std::array<int32_t, 15> outputs{};
        uint32_t output_count = 0;
        uint32_t cycles = 0;
        if (!health) {
            top.p_start.set<uint32_t>(0);
            rising_edge(top);
            settle_low(top);
        } else {
            top.p_start.set<uint32_t>(1);
            rising_edge(top);
            settle_low(top);
            top.p_start.set<uint32_t>(0);
            for (cycles = 1; cycles <= 96; ++cycles) {
                rising_edge(top);
                settle_low(top);
                if (top.p_out__valid.get<uint32_t>()) {
                    const uint32_t index = top.p_out__index.get<uint32_t>();
                    if (index >= outputs.size() || index != output_count)
                        throw std::runtime_error("output ordering mismatch");
                    outputs[index] = signed18(top.p_out__value.get<uint32_t>());
                    ++output_count;
                }
                if (top.p_done.get<uint32_t>())
                    break;
            }
            if (cycles > 96 || output_count != 15)
                throw std::runtime_error("student update did not complete exactly");
        }
        std::cout << step << ',' << outcome << ',' << health << ',' << cycles
                  << ',' << hex_wire(top.p_state__debug);
        for (const auto value : outputs)
            std::cout << ',' << value;
        std::cout << '\n';
    }
    return 0;
}
