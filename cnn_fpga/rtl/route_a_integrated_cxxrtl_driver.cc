#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "route_a_integrated_qualification_model.cc"

#pragma pack(push, 1)
struct TraceRow {
    uint8_t family_id;
    uint8_t reset_before;
    uint8_t in_valid;
    uint8_t safe_boundary;
    uint8_t host_commit_valid;
    uint8_t host_commit_bank;
    uint16_t host_commit_version;
    uint8_t cfg_we;
    uint8_t cfg_bank;
    uint8_t cfg_phase;
    uint16_t cfg_address;
    uint32_t cfg_data;
    uint8_t bank0_trusted;
    uint8_t bank1_trusted;
    uint64_t in_word;
    uint8_t posterior_valid;
    uint8_t p_normal;
    uint8_t p_smooth;
    uint8_t p_calibration;
    uint8_t p_burst;
    uint8_t policy_ood_code;
    uint8_t router_boundary;
    uint8_t window_prequential_win;
    uint8_t integrity_fault;
    uint8_t version_fault;
    uint8_t integrity_clear;
    uint8_t leakage_event;
    uint8_t reset_ack;
    uint8_t lkg_bank;
    uint8_t expected_commit_ack;
    uint8_t expected_active_bank;
    uint16_t expected_active_version;
    uint8_t expected_map_valid;
    uint16_t expected_map_address;
    uint32_t expected_map_llr;
    uint8_t expected_out_word[15];
    uint8_t expected_state_word[29];
    uint8_t expected_route_action_word[10];
    uint8_t expected_route_state_word[12];
    uint8_t expected_route_version_word[8];
    uint8_t expected_route_action;
    uint8_t expected_route_reason;
    uint8_t expected_route_selected_bank;
    uint8_t expected_route_commit_pending;
    uint8_t expected_host_commit_blocked;
};
#pragma pack(pop)

static_assert(sizeof(TraceRow) == 131, "Python/C++ trace ABI drifted");

using Top = cxxrtl_design::p_route__a__integrated__qualification__top;

struct Stats {
    uint64_t rows = 0;
    uint64_t mismatches = 0;
    uint64_t undefined_action = 0;
    uint64_t action_crc_errors = 0;
    uint64_t state_crc_errors = 0;
    uint64_t version_crc_errors = 0;
    uint64_t shadow_mutations = 0;
    uint64_t shadow_mutations_detected = 0;
    uint64_t actual_digest = 1469598103934665603ULL;
    uint64_t expected_digest = 1469598103934665603ULL;
};

static void settle_low(Top &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(Top &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

static void reset_top(Top &top) {
    top.p_reset__n.set<uint32_t>(0);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<uint32_t>(1);
}

template <size_t Bits, size_t Bytes>
static std::array<uint8_t, Bytes> wire_bytes(const cxxrtl::wire<Bits> &wire) {
    std::array<uint8_t, Bytes> result{};
    for (size_t chunk_index = 0; chunk_index < wire.curr.chunks; ++chunk_index) {
        const uint32_t chunk = wire.curr.data[chunk_index];
        for (size_t byte_index = 0; byte_index < 4; ++byte_index) {
            const size_t target = chunk_index * 4 + byte_index;
            if (target < Bytes)
                result[target] = static_cast<uint8_t>((chunk >> (8 * byte_index)) & 0xffU);
        }
    }
    return result;
}

static uint32_t bits(const uint8_t *bytes, unsigned offset, unsigned width) {
    uint32_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned source = offset + bit;
        value |= ((bytes[source / 8] >> (source % 8)) & 1U) << bit;
    }
    return value;
}

static uint16_t crc16(const uint8_t *bytes, size_t count) {
    uint16_t crc = 0xffffU;
    for (size_t index = 0; index < count; ++index) {
        crc ^= static_cast<uint16_t>(bytes[index]) << 8;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc & 0x8000U) ? static_cast<uint16_t>((crc << 1) ^ 0x1021U)
                                  : static_cast<uint16_t>(crc << 1);
    }
    return crc;
}

static void fnv_update(uint64_t &digest, const uint8_t *bytes, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        digest ^= bytes[index];
        digest *= 1099511628211ULL;
    }
}

template <typename T>
static void fnv_scalar(uint64_t &digest, T value) {
    fnv_update(digest, reinterpret_cast<const uint8_t *>(&value), sizeof(T));
}

int main(int argc, char **argv) {
    if (argc != 3)
        throw std::runtime_error("usage: route_a_integrated_driver TRACE.bin FAMILY_ID");
    const int selected_family = std::stoi(argv[2]);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
        throw std::runtime_error("unable to open trace");

    Top top;
    top.p_reset__n.set<uint32_t>(1);
    top.p_in__valid.set<uint32_t>(0);
    top.p_in__word.set<uint64_t>(0);
    top.p_safe__boundary.set<uint32_t>(1);
    top.p_host__commit__valid.set<uint32_t>(0);
    top.p_host__commit__bank.set<uint32_t>(0);
    top.p_host__commit__version.set<uint32_t>(0);
    top.p_cfg__we.set<uint32_t>(0);
    top.p_cfg__bank.set<uint32_t>(0);
    top.p_cfg__phase.set<uint32_t>(0);
    top.p_cfg__address.set<uint32_t>(0);
    top.p_cfg__data.set<uint32_t>(0);
    top.p_bank0__trusted.set<uint32_t>(1);
    top.p_bank1__trusted.set<uint32_t>(1);
    top.p_posterior__valid.set<uint32_t>(0);
    top.p_p__normal.set<uint32_t>(255);
    top.p_p__smooth.set<uint32_t>(0);
    top.p_p__calibration.set<uint32_t>(0);
    top.p_p__burst.set<uint32_t>(0);
    top.p_policy__ood__code.set<uint32_t>(0);
    top.p_router__boundary.set<uint32_t>(0);
    top.p_window__prequential__win.set<uint32_t>(0);
    top.p_integrity__fault.set<uint32_t>(0);
    top.p_version__fault.set<uint32_t>(0);
    top.p_integrity__clear.set<uint32_t>(0);
    top.p_leakage__event.set<uint32_t>(0);
    top.p_reset__ack.set<uint32_t>(0);
    top.p_lkg__bank.set<uint32_t>(0);

    Stats stats;
    uint64_t row_index = 0;
    uint64_t reported = 0;
    TraceRow row{};
    while (input.read(reinterpret_cast<char *>(&row), sizeof(row))) {
        if (row.family_id != selected_family) {
            ++row_index;
            continue;
        }
        if (row.reset_before)
            reset_top(top);
        top.p_in__valid.set<uint32_t>(row.in_valid);
        top.p_in__word.set<uint64_t>(row.in_word);
        top.p_safe__boundary.set<uint32_t>(row.safe_boundary);
        top.p_host__commit__valid.set<uint32_t>(row.host_commit_valid);
        top.p_host__commit__bank.set<uint32_t>(row.host_commit_bank);
        top.p_host__commit__version.set<uint32_t>(row.host_commit_version);
        top.p_cfg__we.set<uint32_t>(row.cfg_we);
        top.p_cfg__bank.set<uint32_t>(row.cfg_bank);
        top.p_cfg__phase.set<uint32_t>(row.cfg_phase);
        top.p_cfg__address.set<uint32_t>(row.cfg_address);
        top.p_cfg__data.set<uint32_t>(row.cfg_data);
        top.p_bank0__trusted.set<uint32_t>(row.bank0_trusted);
        top.p_bank1__trusted.set<uint32_t>(row.bank1_trusted);
        top.p_posterior__valid.set<uint32_t>(row.posterior_valid);
        top.p_p__normal.set<uint32_t>(row.p_normal);
        top.p_p__smooth.set<uint32_t>(row.p_smooth);
        top.p_p__calibration.set<uint32_t>(row.p_calibration);
        top.p_p__burst.set<uint32_t>(row.p_burst);
        top.p_policy__ood__code.set<uint32_t>(row.policy_ood_code);
        top.p_router__boundary.set<uint32_t>(row.router_boundary);
        top.p_window__prequential__win.set<uint32_t>(row.window_prequential_win);
        top.p_integrity__fault.set<uint32_t>(row.integrity_fault);
        top.p_version__fault.set<uint32_t>(row.version_fault);
        top.p_integrity__clear.set<uint32_t>(row.integrity_clear);
        top.p_leakage__event.set<uint32_t>(row.leakage_event);
        top.p_reset__ack.set<uint32_t>(row.reset_ack);
        top.p_lkg__bank.set<uint32_t>(row.lkg_bank);
        rising_edge(top);

        const auto out = wire_bytes<118, 15>(top.p_out__word);
        const auto state = wire_bytes<232, 29>(top.p_state__word);
        const auto action = wire_bytes<80, 10>(top.p_route__action__word);
        const auto route_state = wire_bytes<96, 12>(top.p_route__state__word);
        const auto version = wire_bytes<64, 8>(top.p_route__version__word);
        const uint8_t commit_ack = top.p_commit__ack.get<uint32_t>();
        const uint8_t active_bank = top.p_active__bank__debug.get<uint32_t>();
        const uint16_t active_version = top.p_active__version__debug.get<uint32_t>();
        const uint8_t map_valid = top.p_map__valid__debug.get<uint32_t>();
        const uint16_t map_address = top.p_map__address__debug.get<uint32_t>();
        const uint32_t map_llr = top.p_map__llr__debug.get<uint32_t>();
        const uint8_t route_action = top.p_route__action__debug.get<uint32_t>();
        const uint8_t route_reason = top.p_route__reason__debug.get<uint32_t>();
        const uint8_t selected_bank = top.p_route__selected__bank__debug.get<uint32_t>();
        const uint8_t commit_pending = top.p_route__commit__pending__debug.get<uint32_t>();
        const uint8_t host_commit_blocked = top.p_host__commit__blocked__debug.get<uint32_t>();

        const auto matches = [&](const TraceRow &candidate) {
            return commit_ack == candidate.expected_commit_ack &&
                   active_bank == candidate.expected_active_bank &&
                   active_version == candidate.expected_active_version &&
                   map_valid == candidate.expected_map_valid &&
                   map_address == candidate.expected_map_address &&
                   map_llr == candidate.expected_map_llr &&
                   std::memcmp(out.data(), candidate.expected_out_word, 15) == 0 &&
                   std::memcmp(state.data(), candidate.expected_state_word, 29) == 0 &&
                   std::memcmp(action.data(), candidate.expected_route_action_word, 10) == 0 &&
                   std::memcmp(route_state.data(), candidate.expected_route_state_word, 12) == 0 &&
                   std::memcmp(version.data(), candidate.expected_route_version_word, 8) == 0 &&
                   route_action == candidate.expected_route_action &&
                   route_reason == candidate.expected_route_reason &&
                   selected_bank == candidate.expected_route_selected_bank &&
                   commit_pending == candidate.expected_route_commit_pending &&
                   host_commit_blocked == candidate.expected_host_commit_blocked;
        };

        ++stats.rows;
        if (!matches(row)) {
            ++stats.mismatches;
            if (reported < 16) {
                std::cerr << "mismatch row=" << row_index << " family=" << selected_family
                          << " core=" << unsigned(commit_ack) << '/' << unsigned(active_bank)
                          << '/' << active_version << " policy=" << unsigned(route_action)
                          << '/' << unsigned(route_reason) << '/' << unsigned(selected_bank)
                          << '/' << unsigned(commit_pending)
                          << " fields="
                          << (commit_ack == row.expected_commit_ack) << '/'
                          << (active_bank == row.expected_active_bank) << '/'
                          << (active_version == row.expected_active_version) << '/'
                          << (map_valid == row.expected_map_valid) << '/'
                          << (map_address == row.expected_map_address) << '/'
                          << (map_llr == row.expected_map_llr) << '/'
                          << (std::memcmp(out.data(), row.expected_out_word, 15) == 0) << '/'
                          << (std::memcmp(state.data(), row.expected_state_word, 29) == 0) << '/'
                          << (std::memcmp(action.data(), row.expected_route_action_word, 10) == 0) << '/'
                          << (std::memcmp(route_state.data(), row.expected_route_state_word, 12) == 0) << '/'
                          << (std::memcmp(version.data(), row.expected_route_version_word, 8) == 0) << '/'
                          << (route_action == row.expected_route_action) << '/'
                          << (route_reason == row.expected_route_reason) << '/'
                          << (selected_bank == row.expected_route_selected_bank) << '/'
                          << (commit_pending == row.expected_route_commit_pending) << '/'
                          << (host_commit_blocked == row.expected_host_commit_blocked) << '\n';
                ++reported;
            }
        }
        stats.undefined_action += route_action > 4 || route_reason > 8 || bits(action.data(), 1, 3) > 4;
        stats.action_crc_errors += crc16(action.data(), 8) != bits(action.data(), 64, 16);
        stats.state_crc_errors += crc16(route_state.data(), 10) != bits(route_state.data(), 80, 16);
        stats.version_crc_errors += crc16(version.data(), 6) != bits(version.data(), 48, 16);

        if (stats.rows <= 13) {
            TraceRow mutated = row;
            switch (stats.rows - 1) {
            case 0: mutated.expected_commit_ack ^= 1U; break;
            case 1: mutated.expected_active_version ^= 1U; break;
            case 2: mutated.expected_map_llr ^= 1U; break;
            case 3: mutated.expected_out_word[0] ^= 1U; break;
            case 4: mutated.expected_state_word[0] ^= 1U; break;
            case 5: mutated.expected_route_action_word[0] ^= 1U; break;
            case 6: mutated.expected_route_state_word[0] ^= 1U; break;
            case 7: mutated.expected_route_version_word[0] ^= 1U; break;
            case 8: mutated.expected_route_action ^= 1U; break;
            case 9: mutated.expected_route_reason ^= 1U; break;
            case 10: mutated.expected_route_selected_bank ^= 1U; break;
            case 11: mutated.expected_route_commit_pending ^= 1U; break;
            case 12: mutated.expected_host_commit_blocked ^= 1U; break;
            }
            ++stats.shadow_mutations;
            stats.shadow_mutations_detected += !matches(mutated);
        }

        fnv_scalar(stats.actual_digest, commit_ack);
        fnv_scalar(stats.actual_digest, active_bank);
        fnv_scalar(stats.actual_digest, active_version);
        fnv_scalar(stats.actual_digest, map_valid);
        fnv_scalar(stats.actual_digest, map_address);
        fnv_scalar(stats.actual_digest, map_llr);
        fnv_update(stats.actual_digest, out.data(), out.size());
        fnv_update(stats.actual_digest, state.data(), state.size());
        fnv_update(stats.actual_digest, action.data(), action.size());
        fnv_update(stats.actual_digest, route_state.data(), route_state.size());
        fnv_update(stats.actual_digest, version.data(), version.size());
        fnv_scalar(stats.actual_digest, route_action);
        fnv_scalar(stats.actual_digest, route_reason);
        fnv_scalar(stats.actual_digest, selected_bank);
        fnv_scalar(stats.actual_digest, commit_pending);
        fnv_scalar(stats.actual_digest, host_commit_blocked);

        fnv_scalar(stats.expected_digest, row.expected_commit_ack);
        fnv_scalar(stats.expected_digest, row.expected_active_bank);
        fnv_scalar(stats.expected_digest, row.expected_active_version);
        fnv_scalar(stats.expected_digest, row.expected_map_valid);
        fnv_scalar(stats.expected_digest, row.expected_map_address);
        fnv_scalar(stats.expected_digest, row.expected_map_llr);
        fnv_update(stats.expected_digest, row.expected_out_word, 15);
        fnv_update(stats.expected_digest, row.expected_state_word, 29);
        fnv_update(stats.expected_digest, row.expected_route_action_word, 10);
        fnv_update(stats.expected_digest, row.expected_route_state_word, 12);
        fnv_update(stats.expected_digest, row.expected_route_version_word, 8);
        fnv_scalar(stats.expected_digest, row.expected_route_action);
        fnv_scalar(stats.expected_digest, row.expected_route_reason);
        fnv_scalar(stats.expected_digest, row.expected_route_selected_bank);
        fnv_scalar(stats.expected_digest, row.expected_route_commit_pending);
        fnv_scalar(stats.expected_digest, row.expected_host_commit_blocked);
        ++row_index;
    }

    std::cout << "family_id,rows,mismatches,undefined_action,action_crc_errors,"
                 "state_crc_errors,version_crc_errors,shadow_mutations,"
                 "shadow_mutations_detected,actual_digest,expected_digest\n";
    std::cout << selected_family << ',' << stats.rows << ',' << stats.mismatches << ','
              << stats.undefined_action << ',' << stats.action_crc_errors << ','
              << stats.state_crc_errors << ',' << stats.version_crc_errors << ','
              << stats.shadow_mutations << ',' << stats.shadow_mutations_detected << ','
              << std::hex << std::setw(16) << std::setfill('0') << stats.actual_digest << ','
              << std::setw(16) << stats.expected_digest << std::dec << '\n';
    return stats.mismatches == 0 ? 0 : 2;
}
