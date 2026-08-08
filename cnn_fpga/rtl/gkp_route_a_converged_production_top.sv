// T6.25.2: converged single-mode production top.
//
// Unlike the legacy Route-A qualification top, this module does not expose
// raw bank trust or core configuration enables.  All image writes and commits
// pass through gkp_parameter_bank_manager; the policy overlay supplies only a
// request, and the manager remains the sole CRC/trust/CAS/drain authority.

module gkp_route_a_converged_production_top #(
    parameter integer WORDS_PER_PHASE = 257,
    parameter [7:0] OOD_THRESHOLD_CODE = 8'd192,
    parameter [15:0] MAX_PARAMETER_AGE_CYCLES = 16'd8192,
    parameter [3:0] RETIRED_BANK_DRAIN_CYCLES = 4'd6
) (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          in_valid,
    input  wire [57:0]   in_word,
    input  wire          safe_boundary,

    input  wire          cfg_begin_valid,
    input  wire          cfg_begin_bank,
    input  wire [15:0]   cfg_expected_active_version,
    input  wire [15:0]   cfg_new_image_version,
    input  wire [31:0]   cfg_expected_crc32,
    input  wire          cfg_word_valid,
    input  wire          cfg_word_phase,
    input  wire [8:0]    cfg_word_address,
    input  wire [21:0]   cfg_word_data,
    input  wire          cfg_finalize_valid,
    input  wire          cfg_abort_valid,

    input  wire          host_commit_valid,
    input  wire          host_commit_bank,
    input  wire [15:0]   host_expected_active_version,
    input  wire [15:0]   host_new_activation_version,
    input  wire          commit_cancel_valid,
    input  wire          management_snapshot_request,

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

    output wire          cfg_begin_ack,
    output wire          cfg_word_ack,
    output wire          cfg_finalize_ack,
    output wire          cfg_abort_ack,
    output wire          host_commit_ack,
    output wire          policy_commit_ack,
    output wire          commit_complete,
    output wire          commit_complete_source_policy,
    output wire          commit_cancel_ack,
    output wire          management_snapshot_ack,
    output wire          management_state_valid,
    output wire          management_reject,
    output wire [7:0]    management_reject_reason,
    output wire          management_ready,
    output wire          host_commit_blocked,

    output wire [117:0]  out_word,
    output wire [231:0]  state_word,
    output wire [79:0]   route_action_word,
    output wire [95:0]   route_state_word,
    output wire [63:0]   route_version_word,
    output wire [159:0]  management_state_word,
    output wire          map_valid_debug,
    output wire [21:0]   map_llr_debug,
    output wire [8:0]    map_address_debug,
    output wire [15:0]   active_version_debug,
    output wire          active_bank_debug,
    output wire [2:0]    route_action_debug,
    output wire [3:0]    route_reason_debug,
    output wire          route_selected_bank_debug,
    output wire          route_commit_pending_debug,
    output wire          manager_commit_pending_debug,
    output wire          manager_commit_pending_bank_debug,
    output wire [15:0]   manager_commit_pending_version_debug,
    output wire          manager_commit_pending_source_policy_debug,
    output wire          cfg_session_active_debug,
    output wire          cfg_staged_bank_debug,
    output wire [9:0]    cfg_word_count_debug,
    output wire          cfg_all_words_received_debug,
    output wire [3:0]    retired_bank_drain_count_debug,
    output wire          bank0_trusted_debug,
    output wire          bank1_trusted_debug,
    output wire [15:0]   bank0_image_version_debug,
    output wire [15:0]   bank1_image_version_debug,
    output wire          core_cfg_we_debug,
    output wire          core_cfg_bank_debug,
    output wire          core_commit_valid_debug,
    output wire          core_commit_bank_debug,
    output wire [15:0]   core_commit_version_debug,
    output wire          effective_commit_valid_debug,
    output wire          effective_commit_source_policy_debug
);

    wire policy_auto_commit_valid;
    wire policy_auto_commit_bank;
    wire [15:0] policy_auto_commit_version;
    wire effective_commit_valid;
    wire effective_commit_source_policy;
    wire effective_commit_bank;
    wire [15:0] effective_expected_active_version;
    wire [15:0] effective_new_activation_version;

    wire manager_commit_request_ack;
    wire manager_commit_request_ack_source_policy;
    wire core_commit_ack;
    wire core_cfg_we;
    wire core_cfg_bank;
    wire core_cfg_phase;
    wire [8:0] core_cfg_address;
    wire [21:0] core_cfg_data;
    wire bank0_trusted;
    wire bank1_trusted;
    wire core_commit_valid;
    wire core_commit_bank;
    wire [15:0] core_commit_version;
    wire [31:0] cfg_running_crc32_unused;
    wire [31:0] cfg_expected_crc32_unused;

    route_a_commit_admission admission (
        .policy_commit_valid(policy_auto_commit_valid),
        .policy_commit_bank(policy_auto_commit_bank),
        .policy_commit_version(policy_auto_commit_version),
        .policy_commit_pending(route_commit_pending_debug),
        .policy_action(route_action_debug),
        .policy_selected_bank(route_selected_bank_debug),
        .host_commit_valid(host_commit_valid),
        .host_commit_bank(host_commit_bank),
        .host_expected_active_version(host_expected_active_version),
        .host_new_activation_version(host_new_activation_version),
        .host_commit_blocked(host_commit_blocked),
        .effective_commit_valid(effective_commit_valid),
        .effective_commit_source_policy(effective_commit_source_policy),
        .effective_commit_bank(effective_commit_bank),
        .effective_expected_active_version(effective_expected_active_version),
        .effective_new_activation_version(effective_new_activation_version)
    );

    gkp_parameter_bank_manager #(
        .WORDS_PER_PHASE(WORDS_PER_PHASE),
        .RETIRED_BANK_DRAIN_CYCLES(RETIRED_BANK_DRAIN_CYCLES)
    ) manager (
        .clk(clk),
        .reset_n(reset_n),
        .safe_boundary(safe_boundary),
        .core_active_bank(active_bank_debug),
        .core_active_version(active_version_debug),
        .core_commit_ack(core_commit_ack),
        .cfg_begin_valid(cfg_begin_valid),
        .cfg_begin_bank(cfg_begin_bank),
        .cfg_expected_active_version(cfg_expected_active_version),
        .cfg_new_image_version(cfg_new_image_version),
        .cfg_expected_crc32(cfg_expected_crc32),
        .cfg_word_valid(cfg_word_valid),
        .cfg_word_phase(cfg_word_phase),
        .cfg_word_address(cfg_word_address),
        .cfg_word_data(cfg_word_data),
        .cfg_finalize_valid(cfg_finalize_valid),
        .cfg_abort_valid(cfg_abort_valid),
        .commit_request_valid(effective_commit_valid),
        .commit_request_source_policy(effective_commit_source_policy),
        .commit_request_bank(effective_commit_bank),
        .commit_expected_active_version(effective_expected_active_version),
        .commit_new_activation_version(effective_new_activation_version),
        .commit_cancel_valid(commit_cancel_valid),
        .management_snapshot_request(management_snapshot_request),
        .cfg_begin_ack(cfg_begin_ack),
        .cfg_word_ack(cfg_word_ack),
        .cfg_finalize_ack(cfg_finalize_ack),
        .cfg_abort_ack(cfg_abort_ack),
        .commit_request_ack(manager_commit_request_ack),
        .commit_request_ack_source_policy(manager_commit_request_ack_source_policy),
        .commit_complete(commit_complete),
        .commit_complete_source_policy(commit_complete_source_policy),
        .commit_cancel_ack(commit_cancel_ack),
        .management_snapshot_ack(management_snapshot_ack),
        .management_state_valid(management_state_valid),
        .management_reject(management_reject),
        .management_reject_reason(management_reject_reason),
        .management_ready(management_ready),
        .core_cfg_we(core_cfg_we),
        .core_cfg_bank(core_cfg_bank),
        .core_cfg_phase(core_cfg_phase),
        .core_cfg_address(core_cfg_address),
        .core_cfg_data(core_cfg_data),
        .core_bank0_trusted(bank0_trusted),
        .core_bank1_trusted(bank1_trusted),
        .core_commit_valid(core_commit_valid),
        .core_commit_bank(core_commit_bank),
        .core_commit_version(core_commit_version),
        .cfg_session_active_debug(cfg_session_active_debug),
        .cfg_staged_bank_debug(cfg_staged_bank_debug),
        .cfg_staged_image_version_debug(),
        .cfg_next_phase_debug(),
        .cfg_next_address_debug(),
        .cfg_word_count_debug(cfg_word_count_debug),
        .cfg_all_words_received_debug(cfg_all_words_received_debug),
        .commit_pending_debug(manager_commit_pending_debug),
        .commit_pending_bank_debug(manager_commit_pending_bank_debug),
        .commit_pending_version_debug(manager_commit_pending_version_debug),
        .commit_pending_source_policy_debug(manager_commit_pending_source_policy_debug),
        .retired_bank_drain_count_debug(retired_bank_drain_count_debug),
        .bank0_image_version_debug(bank0_image_version_debug),
        .bank1_image_version_debug(bank1_image_version_debug),
        .cfg_running_crc32_debug(cfg_running_crc32_unused),
        .cfg_expected_crc32_debug(cfg_expected_crc32_unused),
        .management_state_word(management_state_word)
    );

    assign host_commit_ack = manager_commit_request_ack &&
                             !manager_commit_request_ack_source_policy;
    assign policy_commit_ack = manager_commit_request_ack &&
                               manager_commit_request_ack_source_policy;
    assign bank0_trusted_debug = bank0_trusted;
    assign bank1_trusted_debug = bank1_trusted;
    assign core_cfg_we_debug = core_cfg_we;
    assign core_cfg_bank_debug = core_cfg_bank;
    assign core_commit_valid_debug = core_commit_valid;
    assign core_commit_bank_debug = core_commit_bank;
    assign core_commit_version_debug = core_commit_version;
    assign effective_commit_valid_debug = effective_commit_valid;
    assign effective_commit_source_policy_debug = effective_commit_source_policy;

    gkp_fast_path_core #(
        .OOD_THRESHOLD_CODE(OOD_THRESHOLD_CODE),
        .MAX_PARAMETER_AGE_CYCLES(MAX_PARAMETER_AGE_CYCLES),
        .MAX_TRUSTED_BANK_VERSION(16'hffff)
    ) core (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(in_valid),
        .in_word(in_word),
        .safe_boundary(safe_boundary),
        .commit_valid(core_commit_valid),
        .commit_bank(core_commit_bank),
        .commit_version(core_commit_version),
        .commit_ack(core_commit_ack),
        .cfg_we(core_cfg_we),
        .cfg_bank(core_cfg_bank),
        .cfg_phase(core_cfg_phase),
        .cfg_address(core_cfg_address),
        .cfg_data(core_cfg_data),
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
        .auto_commit_valid(policy_auto_commit_valid),
        .auto_commit_bank(policy_auto_commit_bank),
        .auto_commit_version(policy_auto_commit_version),
        .action_debug(route_action_debug),
        .reason_debug(route_reason_debug),
        .selected_bank_debug(route_selected_bank_debug),
        .commit_pending_debug(route_commit_pending_debug),
        .route_action_word(route_action_word),
        .route_state_word(route_state_word),
        .route_version_word(route_version_word)
    );
endmodule
