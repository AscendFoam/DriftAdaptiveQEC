#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "quantized_gru_workload_model.cc"

static void settle_low(cxxrtl_design::p_quantized__gru__workload__kernel &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(cxxrtl_design::p_quantized__gru__workload__kernel &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

int main() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    cxxrtl_design::p_quantized__gru__workload__kernel top;
    top.p_start.set<uint32_t>(0);
    top.p_activation__seed.set<uint32_t>(0x13579);
    top.p_reset__n.set<uint32_t>(0);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<uint32_t>(1);

    top.p_start.set<uint32_t>(1);
    rising_edge(top);
    settle_low(top);
    top.p_start.set<uint32_t>(0);
    if (!top.p_busy.get<uint32_t>())
        throw std::runtime_error("workload did not enter busy state");

    constexpr uint32_t maximum_cycles = 80000;
    uint32_t cycles_after_start = 0;
    for (; cycles_after_start < maximum_cycles; ++cycles_after_start) {
        rising_edge(top);
        settle_low(top);
        if (top.p_done.get<uint32_t>()) {
            ++cycles_after_start;
            break;
        }
    }
    if (cycles_after_start >= maximum_cycles)
        throw std::runtime_error("workload did not complete within guard");

    const auto weight_macs = top.p_weight__macs__completed.get<uint32_t>();
    const auto biases = top.p_biases__consumed.get<uint32_t>();
    if (weight_macs != 72266 || biases != 587)
        throw std::runtime_error("workload completion counters mismatch");
    if (top.p_busy.get<uint32_t>())
        throw std::runtime_error("busy remained asserted with done");

    std::cout << "cycles_after_start,weight_macs_completed,biases_consumed,done,busy,signature\n";
    std::cout << cycles_after_start << ',' << weight_macs << ',' << biases << ','
              << top.p_done.get<uint32_t>() << ',' << top.p_busy.get<uint32_t>() << ','
              << top.p_signature.get<uint32_t>() << '\n';
    return 0;
}
