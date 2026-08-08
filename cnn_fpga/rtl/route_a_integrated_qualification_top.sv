// T6.7.3 board-independent integration top: frozen production datapath plus
// Route-A integer policy overlay.  This is a CXXRTL qualification surface,
// not a board transport or measured timing claim.

module route_a_integrated_qualification_top (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          in_valid,
    input  wire [57:0]   in_word,
    input  wire          safe_boundary,
    input  wire          host_commit_valid,
    input  wire          host_commit_bank,
    input  wire [15:0]   host_commit_version,
    input  wire          cfg_we,
    input  wire          cfg_bank,
    input  wire          cfg_phase,
    input  wire [8:0]    cfg_address,
    input  wire [21:0]   cfg_data,
    input  wire          bank0_trusted,
    input  wire          bank1_trusted,
    input  wire          posterior_valid,
    input  wire [7:0]    p_normal,
    input  wire [7:0]    p_smooth,
    input  wire [7:0]    p_calibration,
    input  wire [7:0]    p_burst,
    input  wire [7:0]    policy_ood_code,
    input  wire          router_boundary,
    input  wire          window_prequential_win,
    input  wire          integrity_fault,
    input  wire          version_fault,
    input  wire          integrity_clear,
    input  wire          leakage_event,
    input  wire          reset_ack,
    input  wire          lkg_bank,

    output wire          commit_ack,
    output wire [117:0]  out_word,
    output wire [231:0]  state_word,
    output wire          map_valid_debug,
    output wire [21:0]   map_llr_debug,
    output wire [8:0]    map_address_debug,
    output wire [15:0]   active_version_debug,
    output wire          active_bank_debug,
    output wire [2:0]    route_action_debug,
    output wire [3:0]    route_reason_debug,
    output wire          route_selected_bank_debug,
    output wire          route_commit_pending_debug,
    output wire          host_commit_blocked_debug,
    output wire [79:0]   route_action_word,
    output wire [95:0]   route_state_word,
    output wire [63:0]   route_version_word
);

    wire auto_commit_valid;
    wire auto_commit_bank;
    wire [15:0] auto_commit_version;
    wire host_commit_allowed = host_commit_valid &&
                               (route_action_debug == 3'd0) &&
                               !route_commit_pending_debug &&
                               (host_commit_bank == route_selected_bank_debug);
    assign host_commit_blocked_debug = host_commit_valid && !host_commit_allowed;
    wire effective_commit_valid = auto_commit_valid || host_commit_allowed;
    wire effective_commit_bank = auto_commit_valid ? auto_commit_bank : host_commit_bank;
    wire [15:0] effective_commit_version = auto_commit_valid ?
                                                  auto_commit_version : host_commit_version;

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
        .commit_valid(effective_commit_valid),
        .commit_bank(effective_commit_bank),
        .commit_version(effective_commit_version),
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

    route_a_policy_overlay policy (
        .clk(clk),
        .reset_n(reset_n),
        .sample_valid(in_valid),
        .safe_boundary(safe_boundary),
        .posterior_valid(posterior_valid),
        .p_normal(p_normal),
        .p_smooth(p_smooth),
        .p_calibration(p_calibration),
        .p_burst(p_burst),
        .ood_code(policy_ood_code),
        .router_boundary(router_boundary),
        .window_prequential_win(window_prequential_win),
        .integrity_fault(integrity_fault),
        .version_fault(version_fault),
        .integrity_clear(integrity_clear),
        .leakage_event(leakage_event),
        .reset_ack(reset_ack),
        .lkg_bank(lkg_bank),
        .active_bank(active_bank_debug),
        .active_version(active_version_debug),
        .core_out_word(out_word),
        .auto_commit_valid(auto_commit_valid),
        .auto_commit_bank(auto_commit_bank),
        .auto_commit_version(auto_commit_version),
        .action_debug(route_action_debug),
        .reason_debug(route_reason_debug),
        .selected_bank_debug(route_selected_bank_debug),
        .commit_pending_debug(route_commit_pending_debug),
        .route_action_word(route_action_word),
        .route_state_word(route_state_word),
        .route_version_word(route_version_word)
    );

endmodule
