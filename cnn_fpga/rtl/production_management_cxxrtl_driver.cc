#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gkp_fast_path_production_model.cc"

template <size_t Bits>
static std::string hex_wire(const cxxrtl::wire<Bits> &wire) {
    const std::string encoded = wire.curr.str();
    const auto separator = encoded.find('\'');
    if (separator == std::string::npos)
        throw std::runtime_error("unexpected CXXRTL value encoding");
    return encoded.substr(separator + 1);
}

using ProductionTop = cxxrtl_design::p_gkp__fast__path__production__top;

static void settle_low(ProductionTop &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(ProductionTop &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

int main() {
    ProductionTop top;
    top.p_in__valid.set<uint32_t>(0);
    top.p_in__word.set<uint64_t>(0);
    top.p_safe__boundary.set<uint32_t>(1);
    top.p_cfg__begin__valid.set<uint32_t>(0);
    top.p_cfg__begin__bank.set<uint32_t>(0);
    top.p_cfg__expected__active__version.set<uint32_t>(0);
    top.p_cfg__new__version.set<uint32_t>(0);
    top.p_cfg__expected__crc32.set<uint32_t>(0);
    top.p_cfg__word__valid.set<uint32_t>(0);
    top.p_cfg__word__phase.set<uint32_t>(0);
    top.p_cfg__word__address.set<uint32_t>(0);
    top.p_cfg__word__data.set<uint32_t>(0);
    top.p_cfg__finalize__valid.set<uint32_t>(0);
    top.p_cfg__abort__valid.set<uint32_t>(0);
    top.p_commit__request__valid.set<uint32_t>(0);
    top.p_commit__request__bank.set<uint32_t>(0);
    top.p_commit__expected__active__version.set<uint32_t>(0);
    top.p_commit__new__version.set<uint32_t>(0);
    top.p_commit__cancel__valid.set<uint32_t>(0);
    top.p_management__snapshot__request.set<uint32_t>(0);
    top.p_reset__n.set<uint32_t>(0);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<uint32_t>(1);

    std::cout << "cycle,cfg_begin_ack,cfg_word_ack,cfg_finalize_ack,cfg_abort_ack,"
                 "commit_request_ack,commit_complete,commit_cancel_ack,management_snapshot_ack,"
                 "management_state_valid,management_reject,management_reject_reason,"
                 "cfg_session_active,commit_pending,management_snapshot_busy,active_bank,"
                 "active_version,management_state_word_hex\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        uint64_t cycle = 0;
        uint32_t safe_boundary = 1;
        uint32_t cfg_begin_valid = 0, cfg_begin_bank = 0;
        uint32_t cfg_expected_active_version = 0, cfg_new_version = 0;
        std::string cfg_expected_crc32_hex;
        uint32_t cfg_word_valid = 0, cfg_word_phase = 0, cfg_word_address = 0;
        std::string cfg_word_data_hex;
        uint32_t cfg_finalize_valid = 0, cfg_abort_valid = 0;
        uint32_t commit_request_valid = 0, commit_request_bank = 0;
        uint32_t commit_expected_active_version = 0, commit_new_version = 0;
        uint32_t commit_cancel_valid = 0;
        uint32_t management_snapshot_request = 0;

        if (!(fields >> cycle >> safe_boundary >>
              cfg_begin_valid >> cfg_begin_bank >> cfg_expected_active_version >>
              cfg_new_version >> cfg_expected_crc32_hex >>
              cfg_word_valid >> cfg_word_phase >> cfg_word_address >> cfg_word_data_hex >>
              cfg_finalize_valid >> cfg_abort_valid >>
              commit_request_valid >> commit_request_bank >>
              commit_expected_active_version >> commit_new_version >>
              commit_cancel_valid >> management_snapshot_request))
            throw std::runtime_error("invalid management trace input: " + line);

        const uint32_t cfg_expected_crc32 =
            static_cast<uint32_t>(std::stoull(cfg_expected_crc32_hex, nullptr, 16));
        const uint32_t cfg_word_data =
            static_cast<uint32_t>(std::stoull(cfg_word_data_hex, nullptr, 16));

        top.p_safe__boundary.set<uint32_t>(safe_boundary);
        top.p_cfg__begin__valid.set<uint32_t>(cfg_begin_valid);
        top.p_cfg__begin__bank.set<uint32_t>(cfg_begin_bank);
        top.p_cfg__expected__active__version.set<uint32_t>(cfg_expected_active_version);
        top.p_cfg__new__version.set<uint32_t>(cfg_new_version);
        top.p_cfg__expected__crc32.set<uint32_t>(cfg_expected_crc32);
        top.p_cfg__word__valid.set<uint32_t>(cfg_word_valid);
        top.p_cfg__word__phase.set<uint32_t>(cfg_word_phase);
        top.p_cfg__word__address.set<uint32_t>(cfg_word_address);
        top.p_cfg__word__data.set<uint32_t>(cfg_word_data);
        top.p_cfg__finalize__valid.set<uint32_t>(cfg_finalize_valid);
        top.p_cfg__abort__valid.set<uint32_t>(cfg_abort_valid);
        top.p_commit__request__valid.set<uint32_t>(commit_request_valid);
        top.p_commit__request__bank.set<uint32_t>(commit_request_bank);
        top.p_commit__expected__active__version.set<uint32_t>(commit_expected_active_version);
        top.p_commit__new__version.set<uint32_t>(commit_new_version);
        top.p_commit__cancel__valid.set<uint32_t>(commit_cancel_valid);
        top.p_management__snapshot__request.set<uint32_t>(management_snapshot_request);

        rising_edge(top);
        std::cout << cycle << ','
                  << top.p_cfg__begin__ack.get<uint32_t>() << ','
                  << top.p_cfg__word__ack.get<uint32_t>() << ','
                  << top.p_cfg__finalize__ack.get<uint32_t>() << ','
                  << top.p_cfg__abort__ack.get<uint32_t>() << ','
                  << top.p_commit__request__ack.get<uint32_t>() << ','
                  << top.p_commit__complete.get<uint32_t>() << ','
                  << top.p_commit__cancel__ack.get<uint32_t>() << ','
                  << top.p_management__snapshot__ack.get<uint32_t>() << ','
                  << top.p_management__state__valid.get<uint32_t>() << ','
                  << top.p_management__reject.get<uint32_t>() << ','
                  << top.p_management__reject__reason.get<uint32_t>() << ','
                  << top.p_cfg__session__active__debug.get<uint32_t>() << ','
                  << top.p_commit__pending__debug.get<uint32_t>() << ','
                  << top.p_management__snapshot__busy__debug.get<uint32_t>() << ','
                  << top.p_active__bank__debug.get<uint32_t>() << ','
                  << top.p_active__version__debug.get<uint32_t>() << ','
                  << hex_wire(top.p_management__state__word) << '\n';
        settle_low(top);
    }
    return 0;
}
