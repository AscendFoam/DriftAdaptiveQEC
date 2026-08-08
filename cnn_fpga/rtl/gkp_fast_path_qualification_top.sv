// T6.2.2 board-independent qualification wrapper.  It exposes the production
// core parameters and raw synchronous management pins to a high-throughput
// CXXRTL verifier.  It is neither a serial transport nor a synthesis harness.

module gkp_fast_path_qualification_top (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          in_valid,
    input  wire [57:0]   in_word,
    input  wire          safe_boundary,
    input  wire          commit_valid,
    input  wire          commit_bank,
    input  wire [15:0]   commit_version,
    output wire          commit_ack,
    input  wire          cfg_we,
    input  wire          cfg_bank,
    input  wire          cfg_phase,
    input  wire [8:0]    cfg_address,
    input  wire [21:0]   cfg_data,
    input  wire          bank0_trusted,
    input  wire          bank1_trusted,
    output wire [117:0]  out_word,
    output wire [231:0]  state_word,
    output wire          map_valid_debug,
    output wire [21:0]   map_llr_debug,
    output wire [8:0]    map_address_debug,
    output wire [15:0]   active_version_debug,
    output wire          active_bank_debug
);

    gkp_fast_path_core #(
        .OOD_THRESHOLD_CODE(8'd192),
        .MAX_PARAMETER_AGE_CYCLES(16'd8192),
        .MAX_TRUSTED_BANK_VERSION(16'hffff)
    ) core (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(in_valid),
        .in_word(in_word),
        .safe_boundary(safe_boundary),
        .commit_valid(commit_valid),
        .commit_bank(commit_bank),
        .commit_version(commit_version),
        .commit_ack(commit_ack),
        .cfg_we(cfg_we),
        .cfg_bank(cfg_bank),
        .cfg_phase(cfg_phase),
        .cfg_address(cfg_address),
        .cfg_data(cfg_data),
        .bank0_trusted(bank0_trusted),
        .bank1_trusted(bank1_trusted),
        .out_word(out_word),
        .state_word(state_word),
        .map_valid_debug(map_valid_debug),
        .map_llr_debug(map_llr_debug),
        .map_address_debug(map_address_debug),
        .active_version_debug(active_version_debug),
        .active_bank_debug(active_bank_debug)
    );

endmodule
