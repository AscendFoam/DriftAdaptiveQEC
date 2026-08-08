#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gkp_fast_path_model.cc"

template <size_t Bits>
static std::string hex_wire(const cxxrtl::wire<Bits> &wire) {
    const std::string encoded = wire.curr.str();
    const auto separator = encoded.find('\'');
    if (separator == std::string::npos)
        throw std::runtime_error("unexpected CXXRTL value encoding");
    return encoded.substr(separator + 1);
}

static void settle_low(cxxrtl_design::p_gkp__fast__path__core &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(cxxrtl_design::p_gkp__fast__path__core &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

int main() {
    cxxrtl_design::p_gkp__fast__path__core top;

    top.p_in__valid.set<uint32_t>(0);
    top.p_in__word.set<uint64_t>(0);
    top.p_safe__boundary.set<uint32_t>(1);
    top.p_commit__valid.set<uint32_t>(0);
    top.p_commit__bank.set<uint32_t>(0);
    top.p_commit__version.set<uint32_t>(0);
    top.p_cfg__we.set<uint32_t>(0);
    top.p_cfg__bank.set<uint32_t>(0);
    top.p_cfg__phase.set<uint32_t>(0);
    top.p_cfg__address.set<uint32_t>(0);
    top.p_cfg__data.set<uint32_t>(0);
    top.p_bank0__trusted.set<uint32_t>(1);
    top.p_bank1__trusted.set<uint32_t>(1);
    top.p_reset__n.set<uint32_t>(0);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<uint32_t>(1);

    std::cout << "cycle,input_valid,input_word_hex,safe_boundary,commit_valid,"
                 "commit_bank,commit_version,commit_ack,active_version,map_valid,"
                 "map_address,map_llr_twos,out_word_hex,state_word_hex\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        uint64_t cycle = 0;
        uint32_t input_valid = 0;
        std::string input_hex;
        uint32_t safe_boundary = 0;
        uint32_t commit_valid = 0;
        uint32_t commit_bank = 0;
        uint32_t commit_version = 0;
        uint32_t bank0_trusted = 1;
        uint32_t bank1_trusted = 1;
        if (!(fields >> cycle >> input_valid >> input_hex >> safe_boundary >>
              commit_valid >> commit_bank >> commit_version >>
              bank0_trusted >> bank1_trusted))
            throw std::runtime_error("invalid trace-driver input line: " + line);

        const uint64_t input_word = std::stoull(input_hex, nullptr, 16);
        if (input_word >= (uint64_t{1} << 58))
            throw std::runtime_error("input word exceeds 58 bits");
        top.p_in__valid.set<uint32_t>(input_valid);
        top.p_in__word.set<uint64_t>(input_word);
        top.p_safe__boundary.set<uint32_t>(safe_boundary);
        top.p_commit__valid.set<uint32_t>(commit_valid);
        top.p_commit__bank.set<uint32_t>(commit_bank);
        top.p_commit__version.set<uint32_t>(commit_version);
        top.p_bank0__trusted.set<uint32_t>(bank0_trusted);
        top.p_bank1__trusted.set<uint32_t>(bank1_trusted);

        rising_edge(top);
        std::cout << cycle << ',' << input_valid << ',' << input_hex << ','
                  << safe_boundary << ',' << commit_valid << ',' << commit_bank << ','
                  << commit_version << ','
                  << top.p_commit__ack.get<uint32_t>() << ','
                  << top.p_active__version__debug.get<uint32_t>() << ','
                  << top.p_map__valid__debug.get<uint32_t>() << ','
                  << top.p_map__address__debug.get<uint32_t>() << ','
                  << top.p_map__llr__debug.get<uint32_t>() << ','
                  << hex_wire(top.p_out__word) << ','
                  << hex_wire(top.p_state__word) << '\n';
        settle_low(top);
    }
    return 0;
}
