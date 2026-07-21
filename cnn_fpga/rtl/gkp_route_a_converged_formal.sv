// T6.25.2 compositional formal harness for the converged Route-A production top.
//
// The parameter-bank image is reduced to two words per phase so a complete
// CRC transaction is reachable in a short SAT trace.  No safety property is
// weakened by the reduction: address order, inactive-bank isolation, trust,
// compare-and-swap, activation monotonicity, cancellation and drain guards are
// independent of the production image depth.  The real core commit handshake
// is represented by its documented acceptance contract; gkp_fast_path_core is
// checked separately for the six-cycle fail-closed data path.

module gkp_route_a_converged_formal #(
    parameter CORE_INITIAL_BANK = 1'b0,
    parameter [15:0] CORE_INITIAL_VERSION = 16'd0
) (
    input wire clk,
    output wire prop_all_state_management_guards,
    output wire prop_all_state_admission_guards,
    output wire prop_all_state_policy_output_guards,
    output reg cover_cfg_write,
    output reg cover_cfg_finalize_success,
    output reg cover_crc_reject,
    output reg cover_commit_admitted,
    output reg cover_active_switch,
    output reg cover_lkg_reactivation,
    output reg cover_cancel,
    output reg cover_drain_reject,
    output reg cover_near_wrap_reject,
    output reg cover_host_blocked,
    output reg cover_policy_priority,
    output reg cover_integrity_rollback,
    output reg cover_conflict_reject
);
    localparam [2:0] ACTION_OPEN               = 3'd0;
    localparam [2:0] ACTION_INTEGRITY_ROLLBACK = 3'd4;

    (* anyseq *) reg reset_n;
    (* anyseq *) reg safe_boundary;
    (* anyseq *) reg cfg_begin_valid;
    (* anyseq *) reg cfg_begin_bank;
    (* anyseq *) reg [15:0] cfg_expected_active_version;
    (* anyseq *) reg [15:0] cfg_new_image_version;
    (* anyseq *) reg [31:0] cfg_expected_crc32;
    (* anyseq *) reg cfg_word_valid;
    (* anyseq *) reg cfg_word_phase;
    (* anyseq *) reg [8:0] cfg_word_address;
    (* anyseq *) reg [21:0] cfg_word_data;
    (* anyseq *) reg cfg_finalize_valid;
    (* anyseq *) reg cfg_abort_valid;
    (* anyseq *) reg host_commit_valid;
    (* anyseq *) reg host_commit_bank;
    (* anyseq *) reg [15:0] host_expected_active_version;
    (* anyseq *) reg [15:0] host_new_activation_version;
    (* anyseq *) reg commit_cancel_valid;
    (* anyseq *) reg management_snapshot_request;

    (* anyseq *) reg posterior_valid;
    (* anyseq *) reg [7:0] p_normal;
    (* anyseq *) reg [7:0] p_smooth;
    (* anyseq *) reg [7:0] p_calibration;
    (* anyseq *) reg [7:0] p_burst;
    (* anyseq *) reg [7:0] policy_ood_code;
    (* anyseq *) reg router_boundary;
    (* anyseq *) reg window_prequential_win;
    (* anyseq *) reg integrity_fault;
    (* anyseq *) reg version_fault;
    (* anyseq *) reg integrity_clear;
    (* anyseq *) reg leakage_event;
    (* anyseq *) reg reset_ack;
    (* anyseq *) reg lkg_bank;

    reg f_past_valid;
    initial f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
    end
    always @* begin
        if (!f_past_valid)
            assume(!reset_n);
        else
            assume(reset_n);
    end

    wire policy_auto_commit_valid;
    wire policy_auto_commit_bank;
    wire [15:0] policy_auto_commit_version;
    wire [2:0] policy_action;
    wire [3:0] policy_reason;
    wire policy_selected_bank;
    wire policy_commit_pending;
    wire [79:0] route_action_word;
    wire [95:0] route_state_word;
    wire [63:0] route_version_word;

    wire host_commit_blocked;
    wire effective_commit_valid;
    wire effective_commit_source_policy;
    wire effective_commit_bank;
    wire [15:0] effective_expected_active_version;
    wire [15:0] effective_new_activation_version;

    wire cfg_begin_ack;
    wire cfg_word_ack;
    wire cfg_finalize_ack;
    wire cfg_abort_ack;
    wire commit_request_ack;
    wire commit_request_ack_source_policy;
    wire commit_complete;
    wire commit_complete_source_policy;
    wire commit_cancel_ack;
    wire management_snapshot_ack;
    wire management_state_valid;
    wire management_reject;
    wire [7:0] management_reject_reason;
    wire management_ready;
    wire core_cfg_we;
    wire core_cfg_bank;
    wire core_cfg_phase;
    wire [8:0] core_cfg_address;
    wire [21:0] core_cfg_data;
    wire core_bank0_trusted;
    wire core_bank1_trusted;
    wire core_commit_valid;
    wire core_commit_bank;
    wire [15:0] core_commit_version;
    wire cfg_session_active;
    wire cfg_staged_bank;
    wire [15:0] cfg_staged_image_version;
    wire cfg_next_phase;
    wire [8:0] cfg_next_address;
    wire [9:0] cfg_word_count;
    wire cfg_all_words_received;
    wire manager_commit_pending;
    wire manager_commit_pending_bank;
    wire [15:0] manager_commit_pending_version;
    wire manager_commit_pending_source_policy;
    wire [3:0] retired_bank_drain_count;
    wire [15:0] bank0_image_version;
    wire [15:0] bank1_image_version;
    wire [31:0] cfg_running_crc32;
    wire [31:0] cfg_staged_expected_crc32;
    wire [159:0] management_state_word;

    // Abstract only the core's already-frozen atomic commit acceptance rule.
    // This keeps the proof state finite while preserving the exact boundary
    // observed by the production manager and policy overlay.
    reg core_active_bank;
    reg [15:0] core_active_version;
    reg core_commit_ack;
    wire abstract_core_accept = core_commit_valid &&
                                (core_commit_bank != core_active_bank) &&
                                (core_active_version != 16'hffff) &&
                                (core_commit_version == core_active_version + 16'd1) &&
                                (core_commit_bank ? core_bank1_trusted :
                                                    core_bank0_trusted);
    always @(posedge clk) begin
        if (!reset_n) begin
            core_active_bank <= CORE_INITIAL_BANK;
            core_active_version <= CORE_INITIAL_VERSION;
            core_commit_ack <= 1'b0;
        end else begin
            core_commit_ack <= abstract_core_accept;
            if (abstract_core_accept) begin
                core_active_bank <= core_commit_bank;
                core_active_version <= core_commit_version;
            end
        end
    end

    route_a_policy_overlay policy (
        .clk(clk),
        .reset_n(reset_n),
        .sample_valid(1'b0),
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
        .active_bank(core_active_bank),
        .active_version(core_active_version),
        .core_out_word(118'd0),
        .auto_commit_valid(policy_auto_commit_valid),
        .auto_commit_bank(policy_auto_commit_bank),
        .auto_commit_version(policy_auto_commit_version),
        .action_debug(policy_action),
        .reason_debug(policy_reason),
        .selected_bank_debug(policy_selected_bank),
        .commit_pending_debug(policy_commit_pending),
        .route_action_word(route_action_word),
        .route_state_word(route_state_word),
        .route_version_word(route_version_word)
    );

    route_a_commit_admission admission (
        .policy_commit_valid(policy_auto_commit_valid),
        .policy_commit_bank(policy_auto_commit_bank),
        .policy_commit_version(policy_auto_commit_version),
        .policy_commit_pending(policy_commit_pending),
        .policy_action(policy_action),
        .policy_selected_bank(policy_selected_bank),
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
        .WORDS_PER_PHASE(2),
        .RETIRED_BANK_DRAIN_CYCLES(4'd6),
        .BANK0_INITIAL_TRUSTED(1'b1),
        .BANK1_INITIAL_TRUSTED(1'b1),
        .BANK0_INITIAL_IMAGE_VERSION(16'd0),
        .BANK1_INITIAL_IMAGE_VERSION(16'd1)
    ) manager (
        .clk(clk),
        .reset_n(reset_n),
        .safe_boundary(safe_boundary),
        .core_active_bank(core_active_bank),
        .core_active_version(core_active_version),
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
        .commit_request_ack(commit_request_ack),
        .commit_request_ack_source_policy(commit_request_ack_source_policy),
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
        .core_bank0_trusted(core_bank0_trusted),
        .core_bank1_trusted(core_bank1_trusted),
        .core_commit_valid(core_commit_valid),
        .core_commit_bank(core_commit_bank),
        .core_commit_version(core_commit_version),
        .cfg_session_active_debug(cfg_session_active),
        .cfg_staged_bank_debug(cfg_staged_bank),
        .cfg_staged_image_version_debug(cfg_staged_image_version),
        .cfg_next_phase_debug(cfg_next_phase),
        .cfg_next_address_debug(cfg_next_address),
        .cfg_word_count_debug(cfg_word_count),
        .cfg_all_words_received_debug(cfg_all_words_received),
        .commit_pending_debug(manager_commit_pending),
        .commit_pending_bank_debug(manager_commit_pending_bank),
        .commit_pending_version_debug(manager_commit_pending_version),
        .commit_pending_source_policy_debug(manager_commit_pending_source_policy),
        .retired_bank_drain_count_debug(retired_bank_drain_count),
        .bank0_image_version_debug(bank0_image_version),
        .bank1_image_version_debug(bank1_image_version),
        .cfg_running_crc32_debug(cfg_running_crc32),
        .cfg_expected_crc32_debug(cfg_staged_expected_crc32),
        .management_state_word(management_state_word)
    );

    wire any_management_request = cfg_begin_valid || cfg_word_valid ||
                                  cfg_finalize_valid || cfg_abort_valid ||
                                  effective_commit_valid || commit_cancel_valid ||
                                  management_snapshot_request;
    wire any_management_outcome = cfg_begin_ack || cfg_word_ack ||
                                  cfg_finalize_ack || cfg_abort_ack ||
                                  commit_request_ack || commit_cancel_ack ||
                                  management_snapshot_ack || management_reject;
    wire isolated_begin_or_commit =
        (cfg_begin_valid ^ effective_commit_valid) &&
        !cfg_word_valid && !cfg_finalize_valid && !cfg_abort_valid &&
        !commit_cancel_valid && !management_snapshot_request;
    wire [2:0] formal_management_request_count =
        {2'b00, cfg_begin_valid} + {2'b00, cfg_word_valid} +
        {2'b00, cfg_finalize_valid} + {2'b00, cfg_abort_valid} +
        {2'b00, effective_commit_valid} + {2'b00, commit_cancel_valid} +
        {2'b00, management_snapshot_request};

    // These two predicates contain only present-state guards.  The runner
    // proves them with every state bit and input unconstrained, separately
    // from the reset-reachable transition proof below.
    assign prop_all_state_management_guards =
        (!core_cfg_we ||
         (cfg_session_active && core_cfg_bank != core_active_bank &&
          core_cfg_address < 9'd2 && !manager_commit_pending)) &&
        (!core_commit_valid ||
         (manager_commit_pending && safe_boundary &&
          core_commit_bank != core_active_bank &&
          core_active_version != 16'hffff &&
          core_commit_version == core_active_version + 16'd1 &&
          (core_commit_bank ? core_bank1_trusted : core_bank0_trusted)));
    assign prop_all_state_admission_guards =
        (!host_commit_blocked || host_commit_valid) &&
        (!(policy_auto_commit_valid && effective_commit_valid) ||
         effective_commit_source_policy) &&
        (!effective_commit_source_policy || policy_auto_commit_valid) &&
        (!effective_commit_valid || effective_commit_source_policy ||
         (host_commit_valid && !policy_auto_commit_valid &&
          !policy_commit_pending && policy_action == ACTION_OPEN));
    assign prop_all_state_policy_output_guards =
        !policy_auto_commit_valid ||
        (policy_commit_pending && safe_boundary &&
         policy_auto_commit_bank != core_active_bank &&
         core_active_version != 16'hffff &&
         policy_auto_commit_version == core_active_version + 16'd1);

    // Independent safety properties.  Assertions deliberately live outside
    // the implementation so source mutations cannot edit the checker too.
    always @* begin
        if (reset_n) begin
            // FORMAL_INDUCTIVE_INVARIANTS_BEGIN
            // Strengthening invariants for induction over all reset-reachable
            // management states.  They are checked, not assumed.
            assert(bank0_image_version != bank1_image_version);
            assert(!cfg_session_active || !manager_commit_pending);
            assert(!cfg_all_words_received || cfg_session_active);
            assert(!cfg_session_active || cfg_staged_bank != core_active_bank);
            assert(!cfg_session_active ||
                   !(cfg_staged_bank ? core_bank1_trusted : core_bank0_trusted));
            assert(!cfg_session_active ||
                   (cfg_staged_image_version > bank0_image_version &&
                    cfg_staged_image_version > bank1_image_version));
            assert(cfg_word_count <= 10'd4);
            assert(!cfg_all_words_received || cfg_word_count == 10'd4);
            assert(!(cfg_session_active && cfg_word_count == 10'd4) ||
                   cfg_all_words_received);
            assert(!cfg_session_active || cfg_next_address < 9'd2);
            if (cfg_session_active && !cfg_all_words_received) begin
                if (cfg_word_count == 10'd0)
                    assert(!cfg_next_phase && cfg_next_address == 9'd0);
                if (cfg_word_count == 10'd1)
                    assert(!cfg_next_phase && cfg_next_address == 9'd1);
                if (cfg_word_count == 10'd2)
                    assert(cfg_next_phase && cfg_next_address == 9'd0);
                if (cfg_word_count == 10'd3)
                    assert(cfg_next_phase && cfg_next_address == 9'd1);
            end
            if (manager_commit_pending) begin
                assert(!cfg_session_active);
                assert(
                    ((manager_commit_pending_bank != core_active_bank) &&
                     (core_active_version != 16'hffff) &&
                     (manager_commit_pending_version == core_active_version + 16'd1) &&
                     (manager_commit_pending_bank ? core_bank1_trusted :
                                                    core_bank0_trusted)) ||
                    (core_commit_ack &&
                     manager_commit_pending_bank == core_active_bank &&
                     manager_commit_pending_version == core_active_version)
                );
            end
            assert(retired_bank_drain_count <= 4'd6);
            assert(policy_action <= ACTION_INTEGRITY_ROLLBACK);
            assert(!commit_request_ack_source_policy || commit_request_ack);
            // FORMAL_INDUCTIVE_INVARIANTS_END
            // FORMAL_PRESENT_STATE_ASSERTIONS_BEGIN
            assert(!core_cfg_we || cfg_session_active);
            assert(!core_cfg_we || (core_cfg_bank != core_active_bank));
            assert(!core_cfg_we || (core_cfg_address < 9'd2));
            assert(!core_cfg_we || !manager_commit_pending);
            assert(!core_commit_valid || manager_commit_pending);
            assert(!core_commit_valid || safe_boundary);
            assert(!core_commit_valid || (core_commit_bank != core_active_bank));
            assert(!core_commit_valid || (core_active_version != 16'hffff));
            assert(!core_commit_valid ||
                   (core_commit_version == core_active_version + 16'd1));
            assert(!core_commit_valid ||
                   (core_commit_bank ? core_bank1_trusted : core_bank0_trusted));
            assert(!policy_auto_commit_valid || policy_commit_pending);
            assert(!policy_auto_commit_valid || safe_boundary);
            assert(!policy_auto_commit_valid ||
                   (policy_auto_commit_bank != core_active_bank));
            assert(!policy_auto_commit_valid ||
                   (core_active_version != 16'hffff));
            assert(!policy_auto_commit_valid ||
                   (policy_auto_commit_version == core_active_version + 16'd1));
            assert(!host_commit_blocked || host_commit_valid);
            assert(!(policy_auto_commit_valid && effective_commit_valid) ||
                   effective_commit_source_policy);
            assert(!effective_commit_source_policy || policy_auto_commit_valid);
            // FORMAL_PRESENT_STATE_ASSERTIONS_END
        end
    end

    // FORMAL_TRANSITION_ASSERTIONS_BEGIN
    always @(posedge clk) begin
        if (f_past_valid && $past(reset_n) && reset_n) begin
            // Atomic active-bank/version state transition.
            if (core_active_bank != $past(core_active_bank)) begin
                assert($past(abstract_core_accept));
                assert(core_active_bank == $past(core_commit_bank));
                assert(core_active_version == $past(core_active_version) + 16'd1);
            end else if (!$past(abstract_core_accept)) begin
                assert(core_active_version == $past(core_active_version));
            end

            // Every admitted management request has a one-cycle explicit
            // outcome; depth-zero backpressure never silently drops a request.
            if ($past(any_management_request))
                assert(any_management_outcome);
            if ($past(formal_management_request_count > 3'd1)) begin
                assert(management_reject);
                assert(management_reject_reason == 8'h01);
            end

            if ($past(integrity_fault)) begin
                assert(policy_action == ACTION_INTEGRITY_ROLLBACK);
                assert(policy_selected_bank == $past(lkg_bank));
            end

            if (cfg_begin_ack) begin
                assert($past(cfg_begin_bank) != $past(core_active_bank));
                assert($past(cfg_expected_active_version) ==
                       $past(core_active_version));
                assert($past(cfg_new_image_version) > $past(bank0_image_version));
                assert($past(cfg_new_image_version) > $past(bank1_image_version));
                assert(!($past(cfg_begin_bank) ? core_bank1_trusted :
                                                   core_bank0_trusted));
            end

            if (cfg_word_ack) begin
                assert($past(core_cfg_we));
                assert($past(cfg_word_phase) == $past(core_cfg_phase));
                assert($past(cfg_word_address) == $past(core_cfg_address));
            end
            if (cfg_finalize_ack) begin
                assert($past(cfg_all_words_received));
                assert($past(cfg_word_count) == 10'd4);
                assert(($past(cfg_running_crc32) ^ 32'hffffffff) ==
                       $past(cfg_staged_expected_crc32));
            end
            if (commit_request_ack) begin
                assert($past(effective_commit_valid));
                assert($past(effective_commit_bank) != $past(core_active_bank));
                assert($past(effective_expected_active_version) ==
                       $past(core_active_version));
                assert($past(effective_new_activation_version) ==
                       $past(core_active_version) + 16'd1);
                assert($past(effective_commit_bank ? core_bank1_trusted :
                                                     core_bank0_trusted));
            end
            if (commit_cancel_ack)
                assert(!manager_commit_pending);
            if (cfg_abort_ack)
                assert(!($past(cfg_staged_bank) ? core_bank1_trusted :
                                                    core_bank0_trusted));
            if ($past(retired_bank_drain_count != 4'd0) &&
                $past(isolated_begin_or_commit)) begin
                assert(management_reject);
                assert(!cfg_begin_ack && !commit_request_ack);
            end
            if ($past(core_active_version == 16'hffff) &&
                $past(isolated_begin_or_commit))
                assert(!cfg_begin_ack && !commit_request_ack);
            if ($past(host_commit_valid && policy_auto_commit_valid)) begin
                assert($past(host_commit_blocked));
                if (commit_request_ack)
                    assert(commit_request_ack_source_policy);
            end
            if (cfg_finalize_ack && $past(cfg_staged_bank)) begin
                assert(core_bank1_trusted);
                assert(bank1_image_version > $past(bank0_image_version));
            end
            if (cfg_finalize_ack && !$past(cfg_staged_bank)) begin
                assert(core_bank0_trusted);
                assert(bank0_image_version > $past(bank1_image_version));
            end
            if (management_reject && (management_reject_reason == 8'h08)) begin
                assert(!($past(cfg_staged_bank) ? core_bank1_trusted :
                                                   core_bank0_trusted));
            end
        end
    end
    // FORMAL_TRANSITION_ASSERTIONS_END

    // Sticky reachability witnesses, queried independently by the runner.
    always @(posedge clk) begin
        if (!reset_n) begin
            cover_cfg_write <= 1'b0;
            cover_cfg_finalize_success <= 1'b0;
            cover_crc_reject <= 1'b0;
            cover_commit_admitted <= 1'b0;
            cover_active_switch <= 1'b0;
            cover_lkg_reactivation <= 1'b0;
            cover_cancel <= 1'b0;
            cover_drain_reject <= 1'b0;
            cover_near_wrap_reject <= 1'b0;
            cover_host_blocked <= 1'b0;
            cover_policy_priority <= 1'b0;
            cover_integrity_rollback <= 1'b0;
            cover_conflict_reject <= 1'b0;
        end else begin
            cover_cfg_write <= cover_cfg_write || cfg_word_ack;
            cover_cfg_finalize_success <= cover_cfg_finalize_success || cfg_finalize_ack;
            cover_crc_reject <= cover_crc_reject ||
                                (management_reject && management_reject_reason == 8'h08);
            cover_commit_admitted <= cover_commit_admitted || commit_request_ack;
            cover_active_switch <= cover_active_switch ||
                                   (f_past_valid && core_active_bank != $past(core_active_bank));
            cover_lkg_reactivation <= cover_lkg_reactivation ||
                (f_past_valid && core_active_bank != $past(core_active_bank) &&
                 ((core_active_bank && bank1_image_version != core_active_version) ||
                  (!core_active_bank && bank0_image_version != core_active_version)));
            cover_cancel <= cover_cancel || commit_cancel_ack;
            cover_drain_reject <= cover_drain_reject ||
                                  (management_reject && management_reject_reason == 8'h05);
            cover_near_wrap_reject <= cover_near_wrap_reject ||
                (f_past_valid && $past(core_active_version == 16'hffff) &&
                 management_reject && management_reject_reason == 8'h04);
            cover_host_blocked <= cover_host_blocked || host_commit_blocked;
            cover_policy_priority <= cover_policy_priority ||
                (policy_auto_commit_valid && host_commit_valid && host_commit_blocked &&
                 effective_commit_source_policy);
            cover_integrity_rollback <= cover_integrity_rollback ||
                (policy_action == ACTION_INTEGRITY_ROLLBACK &&
                 policy_selected_bank == lkg_bank);
            cover_conflict_reject <= cover_conflict_reject ||
                (management_reject && management_reject_reason == 8'h01);
        end
    end
endmodule
