#include "converged_long_model.cc"

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using Top = cxxrtl_design::p_gkp__route__a__converged__production__top;

namespace {

constexpr size_t EXPECTED_BYTES = 148;

#pragma pack(push, 1)
struct TraceRow {
    uint8_t family_id;
    uint8_t reset_before;
    uint8_t in_valid;
    uint64_t in_word;
    uint8_t safe_boundary;
    uint8_t cfg_begin_valid;
    uint8_t cfg_begin_bank;
    uint16_t cfg_expected_active_version;
    uint16_t cfg_new_image_version;
    uint32_t cfg_expected_crc32;
    uint8_t cfg_word_valid;
    uint8_t cfg_word_phase;
    uint16_t cfg_word_address;
    uint32_t cfg_word_data;
    uint8_t cfg_finalize_valid;
    uint8_t cfg_abort_valid;
    uint8_t host_commit_valid;
    uint8_t host_commit_bank;
    uint16_t host_expected_active_version;
    uint16_t host_new_activation_version;
    uint8_t commit_cancel_valid;
    uint8_t management_snapshot_request;
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
    uint8_t expected[EXPECTED_BYTES];
};
#pragma pack(pop)

static_assert(sizeof(TraceRow) == 202, "Python/C++ converged trace ABI drift");

struct FamilyStats {
    uint64_t rows = 0;
    uint64_t mismatches = 0;
    uint64_t output_valid = 0;
    uint64_t route_valid = 0;
    uint64_t map_valid = 0;
    uint64_t undefined_actions = 0;
    uint64_t latency_violations = 0;
    uint64_t map_latency_violations = 0;
    uint64_t route_alignment_violations = 0;
    uint64_t ii1_input_pairs = 0;
    uint64_t ii1_output_pairs = 0;
    uint64_t shadow_mutations = 0;
    uint64_t shadow_mutations_detected = 0;
    uint64_t version_transitions = 0;
    uint64_t silent_version_wraps = 0;
    uint16_t max_active_version = 0;
    uint64_t actual_digest = 1469598103934665603ULL;
    uint64_t expected_digest = 1469598103934665603ULL;
};

void rising_edge(Top &top) {
    top.p_clk.set<bool>(false);
    top.step();
    top.p_clk.set<bool>(true);
    top.step();
}

void settle_low(Top &top) {
    top.p_clk.set<bool>(false);
    top.step();
}

void reset_top(Top &top) {
    top.p_reset__n.set<bool>(false);
    rising_edge(top);
    settle_low(top);
    top.p_reset__n.set<bool>(true);
}

template<size_t Width, size_t Bytes>
std::array<uint8_t, Bytes> wire_bytes(const cxxrtl::wire<Width> &wire) {
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

void fnv_update(uint64_t &digest, const uint8_t *data, size_t size) {
    for (size_t index = 0; index < size; ++index) {
        digest ^= data[index];
        digest *= 1099511628211ULL;
    }
}

template<typename T>
void append_scalar(std::array<uint8_t, EXPECTED_BYTES> &buffer, size_t &offset, T value) {
    for (size_t index = 0; index < sizeof(T); ++index)
        buffer[offset++] = static_cast<uint8_t>((static_cast<uint64_t>(value) >> (8 * index)) & 0xffU);
}

template<size_t N>
void append_array(std::array<uint8_t, EXPECTED_BYTES> &buffer, size_t &offset,
                  const std::array<uint8_t, N> &value) {
    std::memcpy(buffer.data() + offset, value.data(), N);
    offset += N;
}

void apply_inputs(Top &top, const TraceRow &row) {
    top.p_in__valid.set<uint32_t>(row.in_valid);
    top.p_in__word.set<uint64_t>(row.in_word);
    top.p_safe__boundary.set<uint32_t>(row.safe_boundary);
    top.p_cfg__begin__valid.set<uint32_t>(row.cfg_begin_valid);
    top.p_cfg__begin__bank.set<uint32_t>(row.cfg_begin_bank);
    top.p_cfg__expected__active__version.set<uint32_t>(row.cfg_expected_active_version);
    top.p_cfg__new__image__version.set<uint32_t>(row.cfg_new_image_version);
    top.p_cfg__expected__crc32.set<uint32_t>(row.cfg_expected_crc32);
    top.p_cfg__word__valid.set<uint32_t>(row.cfg_word_valid);
    top.p_cfg__word__phase.set<uint32_t>(row.cfg_word_phase);
    top.p_cfg__word__address.set<uint32_t>(row.cfg_word_address);
    top.p_cfg__word__data.set<uint32_t>(row.cfg_word_data);
    top.p_cfg__finalize__valid.set<uint32_t>(row.cfg_finalize_valid);
    top.p_cfg__abort__valid.set<uint32_t>(row.cfg_abort_valid);
    top.p_host__commit__valid.set<uint32_t>(row.host_commit_valid);
    top.p_host__commit__bank.set<uint32_t>(row.host_commit_bank);
    top.p_host__expected__active__version.set<uint32_t>(row.host_expected_active_version);
    top.p_host__new__activation__version.set<uint32_t>(row.host_new_activation_version);
    top.p_commit__cancel__valid.set<uint32_t>(row.commit_cancel_valid);
    top.p_management__snapshot__request.set<uint32_t>(row.management_snapshot_request);
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
}

std::array<uint8_t, EXPECTED_BYTES> actual_bytes(Top &top) {
    std::array<uint8_t, EXPECTED_BYTES> value{};
    size_t offset = 0;
    append_scalar<uint8_t>(value, offset, top.p_cfg__begin__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__word__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__finalize__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__abort__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_host__commit__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_policy__commit__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_commit__complete.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_commit__complete__source__policy.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_commit__cancel__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_management__snapshot__ack.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_management__state__valid.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_management__reject.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_management__reject__reason.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_management__ready.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_host__commit__blocked.get<uint32_t>());
    append_array(value, offset, wire_bytes<118, 15>(top.p_out__word));
    append_array(value, offset, wire_bytes<232, 29>(top.p_state__word));
    append_array(value, offset, wire_bytes<80, 10>(top.p_route__action__word));
    append_array(value, offset, wire_bytes<96, 12>(top.p_route__state__word));
    append_array(value, offset, wire_bytes<64, 8>(top.p_route__version__word));
    append_array(value, offset, wire_bytes<160, 20>(top.p_management__state__word));
    append_scalar<uint8_t>(value, offset, top.p_map__valid__debug.get<uint32_t>());
    append_scalar<uint32_t>(value, offset, top.p_map__llr__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_map__address__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_active__version__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_active__bank__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_route__action__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_route__reason__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_route__selected__bank__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_route__commit__pending__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_manager__commit__pending__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_manager__commit__pending__bank__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_manager__commit__pending__version__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_manager__commit__pending__source__policy__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__session__active__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__staged__bank__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_cfg__word__count__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_cfg__all__words__received__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_retired__bank__drain__count__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_bank0__trusted__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_bank1__trusted__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_bank0__image__version__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_bank1__image__version__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_core__cfg__we__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_core__cfg__bank__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_core__commit__valid__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_core__commit__bank__debug.get<uint32_t>());
    append_scalar<uint16_t>(value, offset, top.p_core__commit__version__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_effective__commit__valid__debug.get<uint32_t>());
    append_scalar<uint8_t>(value, offset, top.p_effective__commit__source__policy__debug.get<uint32_t>());
    if (offset != EXPECTED_BYTES)
        throw std::runtime_error("actual byte pack length mismatch");
    return value;
}

uint32_t get_bits(const uint8_t *bytes, size_t start, size_t width) {
    uint32_t value = 0;
    for (size_t bit = 0; bit < width; ++bit)
        value |= ((bytes[(start + bit) / 8] >> ((start + bit) % 8)) & 1U) << bit;
    return value;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3)
        throw std::runtime_error("usage: converged_long_cxxrtl_driver TRACE.bin FAMILY_ID");
    const int selected_family = std::stoi(argv[2]);
    if (selected_family < 0 || selected_family > 255)
        throw std::runtime_error("FAMILY_ID must fit uint8");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
        throw std::runtime_error("unable to open converged qualification trace");

    Top top;
    TraceRow zero{};
    apply_inputs(top, zero);
    top.p_reset__n.set<bool>(true);
    FamilyStats stats{};
    std::deque<uint64_t> pending_outputs;
    std::deque<uint64_t> pending_maps;
    uint64_t local_cycle = 0;
    uint64_t global_row = 0;
    uint64_t reported = 0;
    bool previous_input = false;
    bool previous_output = false;
    bool have_previous_version = false;
    uint16_t previous_version = 0;
    TraceRow row{};
    while (input.read(reinterpret_cast<char *>(&row), sizeof(row))) {
        if (row.family_id != selected_family) {
            global_row++;
            continue;
        }
        if (row.reset_before) {
            reset_top(top);
            pending_outputs.clear();
            pending_maps.clear();
            previous_input = false;
            previous_output = false;
            have_previous_version = false;
        }
        apply_inputs(top, row);
        if (row.in_valid) {
            pending_outputs.push_back(local_cycle);
            pending_maps.push_back(local_cycle);
        }
        if (previous_input && row.in_valid)
            stats.ii1_input_pairs++;
        rising_edge(top);
        const auto actual = actual_bytes(top);
        const bool exact = std::memcmp(actual.data(), row.expected, EXPECTED_BYTES) == 0;
        stats.rows++;
        if (!exact) {
            stats.mismatches++;
            if (reported < 20) {
                size_t first = 0;
                while (first < EXPECTED_BYTES && actual[first] == row.expected[first])
                    first++;
                std::cerr << "mismatch family=" << selected_family << " cycle=" << local_cycle
                          << " first_expected_byte=" << first;
                if (first < EXPECTED_BYTES)
                    std::cerr << " actual=" << unsigned(actual[first])
                              << " expected=" << unsigned(row.expected[first]);
                std::cerr << '\n';
                reported++;
            }
        }

        const uint8_t out_valid = actual[15] & 1U;
        const uint8_t route_valid = actual[15 + 15 + 29] & 1U;
        const size_t debug_offset = 15 + 15 + 29 + 10 + 12 + 8 + 20;
        const uint8_t map_valid = actual[debug_offset];
        const uint16_t active_version = static_cast<uint16_t>(
            actual[debug_offset + 7] | (uint16_t(actual[debug_offset + 8]) << 8));
        const uint8_t route_action = actual[debug_offset + 10];
        const uint8_t route_reason = actual[debug_offset + 11];
        stats.output_valid += out_valid;
        stats.route_valid += route_valid;
        stats.map_valid += map_valid;
        stats.undefined_actions += int(route_action > 4 || route_reason > 8);
        stats.route_alignment_violations += int(route_valid != out_valid);
        if (out_valid) {
            if (pending_outputs.empty() || local_cycle - pending_outputs.front() != 6)
                stats.latency_violations++;
            else
                pending_outputs.pop_front();
        }
        if (map_valid) {
            if (pending_maps.empty() || local_cycle - pending_maps.front() != 5)
                stats.map_latency_violations++;
            else
                pending_maps.pop_front();
        }
        if (previous_output && out_valid)
            stats.ii1_output_pairs++;
        previous_input = row.in_valid;
        previous_output = out_valid;
        if (have_previous_version && active_version < previous_version)
            stats.silent_version_wraps++;
        if (have_previous_version && active_version != previous_version)
            stats.version_transitions++;
        previous_version = active_version;
        have_previous_version = true;
        if (active_version > stats.max_active_version)
            stats.max_active_version = active_version;

        if (global_row < EXPECTED_BYTES) {
            std::array<uint8_t, EXPECTED_BYTES> mutated{};
            std::memcpy(mutated.data(), row.expected, EXPECTED_BYTES);
            mutated[global_row] ^= 1U;
            stats.shadow_mutations++;
            stats.shadow_mutations_detected +=
                std::memcmp(actual.data(), mutated.data(), EXPECTED_BYTES) != 0;
        }
        fnv_update(stats.actual_digest, actual.data(), actual.size());
        fnv_update(stats.expected_digest, row.expected, EXPECTED_BYTES);
        settle_low(top);
        local_cycle++;
        global_row++;
    }
    if (!input.eof())
        throw std::runtime_error("qualification trace is truncated");

    std::cout << "family_id,rows,mismatches,output_valid,route_valid,map_valid,"
                 "undefined_actions,latency_violations,map_latency_violations,"
                 "route_alignment_violations,ii1_input_pairs,ii1_output_pairs,"
                 "shadow_mutations,shadow_mutations_detected,version_transitions,"
                 "silent_version_wraps,max_active_version,actual_digest,expected_digest\n";
    std::cout << selected_family << ',' << stats.rows << ',' << stats.mismatches << ','
              << stats.output_valid << ',' << stats.route_valid << ',' << stats.map_valid << ','
              << stats.undefined_actions << ',' << stats.latency_violations << ','
              << stats.map_latency_violations << ',' << stats.route_alignment_violations << ','
              << stats.ii1_input_pairs << ',' << stats.ii1_output_pairs << ','
              << stats.shadow_mutations << ',' << stats.shadow_mutations_detected << ','
              << stats.version_transitions << ',' << stats.silent_version_wraps << ','
              << stats.max_active_version << ',' << std::hex << std::setw(16)
              << std::setfill('0') << stats.actual_digest << ',' << std::setw(16)
              << stats.expected_digest << std::dec << '\n';
    return 0;
}
