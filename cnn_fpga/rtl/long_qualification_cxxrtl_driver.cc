#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gkp_fast_path_qualification_model.cc"

#pragma pack(push, 1)
struct TraceRow {
    uint8_t family_id;
    uint8_t reset_before;
    uint8_t in_valid;
    uint8_t safe_boundary;
    uint8_t commit_valid;
    uint8_t commit_bank;
    uint16_t commit_version;
    uint8_t cfg_we;
    uint8_t cfg_bank;
    uint8_t cfg_phase;
    uint16_t cfg_address;
    uint32_t cfg_data;
    uint8_t bank0_trusted;
    uint8_t bank1_trusted;
    uint64_t in_word;
    uint8_t expected_commit_ack;
    uint8_t expected_active_bank;
    uint16_t expected_active_version;
    uint8_t expected_map_valid;
    uint16_t expected_map_address;
    uint32_t expected_map_llr;
    uint8_t expected_out_word[15];
    uint8_t expected_state_word[29];
};
#pragma pack(pop)

static_assert(sizeof(TraceRow) == 82, "Python/C++ trace row contract drifted");

using QualificationTop = cxxrtl_design::p_gkp__fast__path__qualification__top;

struct FamilyStats {
    uint64_t rows = 0;
    uint64_t mismatches = 0;
    uint64_t output_valid = 0;
    uint64_t blocking_fault_outputs = 0;
    uint64_t undefined_actions = 0;
    uint64_t shadow_mutations = 0;
    uint64_t shadow_mutations_detected = 0;
    uint64_t actual_digest = 1469598103934665603ULL;
    uint64_t expected_digest = 1469598103934665603ULL;
};

static void settle_low(QualificationTop &top) {
    top.p_clk.set<uint32_t>(0);
    top.step();
}

static void rising_edge(QualificationTop &top) {
    settle_low(top);
    top.p_clk.set<uint32_t>(1);
    top.step();
}

static void reset_top(QualificationTop &top) {
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

static uint32_t get_bits(const uint8_t *bytes, unsigned offset, unsigned width) {
    uint32_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const unsigned source = offset + bit;
        value |= ((bytes[source / 8] >> (source % 8)) & 1U) << bit;
    }
    return value;
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
    if (argc != 2 && argc != 3)
        throw std::runtime_error("usage: long_qualification_cxxrtl_driver TRACE.bin [FAMILY_ID]");
    const int selected_family = argc == 3 ? std::stoi(argv[2]) : -1;
    if (selected_family > 255)
        throw std::runtime_error("FAMILY_ID must fit uint8");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
        throw std::runtime_error("unable to open binary qualification trace");

    QualificationTop top;
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
    top.p_reset__n.set<uint32_t>(1);

    std::array<FamilyStats, 256> stats{};
    uint64_t row_index = 0;
    uint64_t reported_mismatches = 0;
    TraceRow row{};
    while (input.read(reinterpret_cast<char *>(&row), sizeof(row))) {
        if (selected_family >= 0 && row.family_id != selected_family) {
            row_index++;
            continue;
        }
        if (row.reset_before)
            reset_top(top);
        top.p_in__valid.set<uint32_t>(row.in_valid);
        top.p_in__word.set<uint64_t>(row.in_word);
        top.p_safe__boundary.set<uint32_t>(row.safe_boundary);
        top.p_commit__valid.set<uint32_t>(row.commit_valid);
        top.p_commit__bank.set<uint32_t>(row.commit_bank);
        top.p_commit__version.set<uint32_t>(row.commit_version);
        top.p_cfg__we.set<uint32_t>(row.cfg_we);
        top.p_cfg__bank.set<uint32_t>(row.cfg_bank);
        top.p_cfg__phase.set<uint32_t>(row.cfg_phase);
        top.p_cfg__address.set<uint32_t>(row.cfg_address);
        top.p_cfg__data.set<uint32_t>(row.cfg_data);
        top.p_bank0__trusted.set<uint32_t>(row.bank0_trusted);
        top.p_bank1__trusted.set<uint32_t>(row.bank1_trusted);
        rising_edge(top);

        const auto actual_out = wire_bytes<118, 15>(top.p_out__word);
        const auto actual_state = wire_bytes<232, 29>(top.p_state__word);
        const uint8_t actual_commit_ack = top.p_commit__ack.get<uint32_t>();
        const uint8_t actual_active_bank = top.p_active__bank__debug.get<uint32_t>();
        const uint16_t actual_active_version = top.p_active__version__debug.get<uint32_t>();
        const uint8_t actual_map_valid = top.p_map__valid__debug.get<uint32_t>();
        const uint16_t actual_map_address = top.p_map__address__debug.get<uint32_t>();
        const uint32_t actual_map_llr = top.p_map__llr__debug.get<uint32_t>();

        const auto matches = [&](const TraceRow &candidate) {
            return actual_commit_ack == candidate.expected_commit_ack &&
                   actual_active_bank == candidate.expected_active_bank &&
                   actual_active_version == candidate.expected_active_version &&
                   actual_map_valid == candidate.expected_map_valid &&
                   actual_map_address == candidate.expected_map_address &&
                   actual_map_llr == candidate.expected_map_llr &&
                   std::memcmp(actual_out.data(), candidate.expected_out_word, 15) == 0 &&
                   std::memcmp(actual_state.data(), candidate.expected_state_word, 29) == 0;
        };
        const bool exact = matches(row);

        FamilyStats &family = stats[row.family_id];
        family.rows++;
        if (!exact) {
            family.mismatches++;
            if (reported_mismatches < 20) {
                std::cerr << "mismatch row=" << row_index
                          << " family=" << unsigned(row.family_id)
                          << " scalar="
                          << unsigned(actual_commit_ack) << '/'
                          << unsigned(actual_active_bank) << '/'
                          << actual_active_version << '/'
                          << unsigned(actual_map_valid) << '/'
                          << actual_map_address << '/'
                          << actual_map_llr << '\n';
                reported_mismatches++;
            }
        }

        // Comparator self-test: perturb one independently packed expected field
        // on the first eight trace rows.  A real comparator must reject every
        // shadow mutation without requiring a second million-cycle simulation.
        if (row_index < 8) {
            TraceRow mutated = row;
            switch (row_index) {
            case 0: mutated.expected_commit_ack ^= 1U; break;
            case 1: mutated.expected_active_bank ^= 1U; break;
            case 2: mutated.expected_active_version ^= 1U; break;
            case 3: mutated.expected_map_valid ^= 1U; break;
            case 4: mutated.expected_map_address ^= 1U; break;
            case 5: mutated.expected_map_llr ^= 1U; break;
            case 6: mutated.expected_out_word[0] ^= 1U; break;
            case 7: mutated.expected_state_word[0] ^= 1U; break;
            default: break;
            }
            family.shadow_mutations++;
            family.shadow_mutations_detected += !matches(mutated);
        }

        const uint32_t output_valid = get_bits(actual_out.data(), 0, 1);
        const uint32_t mode = get_bits(actual_out.data(), 1, 3);
        const uint32_t action = get_bits(actual_out.data(), 7, 2);
        const uint32_t health = get_bits(actual_out.data(), 47, 3);
        const uint32_t fault_mask = get_bits(actual_out.data(), 50, 14);
        family.output_valid += output_valid;
        family.blocking_fault_outputs += output_valid && (fault_mask & 0x1fffU);
        family.undefined_actions += output_valid && (mode > 5 || action > 2 || health > 4);

        fnv_scalar(family.actual_digest, actual_commit_ack);
        fnv_scalar(family.actual_digest, actual_active_bank);
        fnv_scalar(family.actual_digest, actual_active_version);
        fnv_scalar(family.actual_digest, actual_map_valid);
        fnv_scalar(family.actual_digest, actual_map_address);
        fnv_scalar(family.actual_digest, actual_map_llr);
        fnv_update(family.actual_digest, actual_out.data(), actual_out.size());
        fnv_update(family.actual_digest, actual_state.data(), actual_state.size());
        fnv_scalar(family.expected_digest, row.expected_commit_ack);
        fnv_scalar(family.expected_digest, row.expected_active_bank);
        fnv_scalar(family.expected_digest, row.expected_active_version);
        fnv_scalar(family.expected_digest, row.expected_map_valid);
        fnv_scalar(family.expected_digest, row.expected_map_address);
        fnv_scalar(family.expected_digest, row.expected_map_llr);
        fnv_update(family.expected_digest, row.expected_out_word, 15);
        fnv_update(family.expected_digest, row.expected_state_word, 29);
        settle_low(top);
        row_index++;
    }
    if (!input.eof())
        throw std::runtime_error("qualification trace is truncated");

    std::cout << "family_id,rows,mismatches,output_valid,blocking_fault_outputs,"
                 "undefined_actions,shadow_mutations,shadow_mutations_detected,"
                 "actual_digest,expected_digest\n";
    for (size_t family_id = 0; family_id < stats.size(); ++family_id) {
        const FamilyStats &family = stats[family_id];
        if (!family.rows)
            continue;
        std::cout << family_id << ',' << family.rows << ',' << family.mismatches << ','
                  << family.output_valid << ',' << family.blocking_fault_outputs << ','
                  << family.undefined_actions << ',' << family.shadow_mutations << ','
                  << family.shadow_mutations_detected << ',' << std::hex << std::setw(16)
                  << std::setfill('0') << family.actual_digest << ',' << std::setw(16)
                  << family.expected_digest << std::dec << '\n';
    }
    return 0;
}
