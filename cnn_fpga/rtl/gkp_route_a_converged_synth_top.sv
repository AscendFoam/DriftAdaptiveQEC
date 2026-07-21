// T6.25.4 small-pin activity/observability harness for synthesis and P&R.
//
// The exact T6.25.2 production top remains the implementation under test.
// This wrapper does not add transport, raw trust/config bypasses or a second
// decoder.  It supplies dynamic activity to every high-level input and folds
// every public output bit through registered lanes so synthesis cannot prune
// the manager, policy, CRC, event/fallback or MAP paths as constants.

module gkp_route_a_converged_synth_top (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] seed_in,
    output wire [7:0] status_out
);
    reg [31:0] lfsr;
    reg [15:0] cycle_counter;
    reg [41:0] input_payload;
    reg [31:0] fold [0:28];
    reg [31:0] signature [0:7];
    integer index;

    wire [57:0] in_word;
    wire safe_boundary = ~cycle_counter[0];
    wire [8:0] safe_address =
        (lfsr[8:0] <= 9'd256) ? lfsr[8:0] : (lfsr[8:0] - 9'd257);
    wire [7:0] p_normal = {2'b00, lfsr[5:0]};
    wire [7:0] p_smooth = {2'b00, lfsr[13:8]};
    wire [7:0] p_calibration = {2'b00, lfsr[21:16]};
    wire [7:0] p_burst = 8'd255 - p_normal - p_smooth - p_calibration;

    wire cfg_begin_valid = cycle_counter[11:0] == 12'h100;
    wire cfg_word_valid = cycle_counter[5:0] == 6'h21;
    wire cfg_finalize_valid = cycle_counter[11:0] == 12'h300;
    wire cfg_abort_valid = cycle_counter[11:0] == 12'h400;
    wire host_commit_valid = cycle_counter[11:0] == 12'h500;
    wire commit_cancel_valid = cycle_counter[11:0] == 12'h600;
    wire management_snapshot_request = cycle_counter[11:0] == 12'h700;

    wire cfg_begin_ack;
    wire cfg_word_ack;
    wire cfg_finalize_ack;
    wire cfg_abort_ack;
    wire host_commit_ack;
    wire policy_commit_ack;
    wire commit_complete;
    wire commit_complete_source_policy;
    wire commit_cancel_ack;
    wire management_snapshot_ack;
    wire management_state_valid;
    wire management_reject;
    wire [7:0] management_reject_reason;
    wire management_ready;
    wire host_commit_blocked;
    wire [117:0] out_word;
    wire [231:0] state_word;
    wire [79:0] route_action_word;
    wire [95:0] route_state_word;
    wire [63:0] route_version_word;
    wire [159:0] management_state_word;
    wire map_valid_debug;
    wire [21:0] map_llr_debug;
    wire [8:0] map_address_debug;
    wire [15:0] active_version_debug;
    wire active_bank_debug;
    wire [2:0] route_action_debug;
    wire [3:0] route_reason_debug;
    wire route_selected_bank_debug;
    wire route_commit_pending_debug;
    wire manager_commit_pending_debug;
    wire manager_commit_pending_bank_debug;
    wire [15:0] manager_commit_pending_version_debug;
    wire manager_commit_pending_source_policy_debug;
    wire cfg_session_active_debug;
    wire cfg_staged_bank_debug;
    wire [9:0] cfg_word_count_debug;
    wire cfg_all_words_received_debug;
    wire [3:0] retired_bank_drain_count_debug;
    wire bank0_trusted_debug;
    wire bank1_trusted_debug;
    wire [15:0] bank0_image_version_debug;
    wire [15:0] bank1_image_version_debug;
    wire core_cfg_we_debug;
    wire core_cfg_bank_debug;
    wire core_commit_valid_debug;
    wire core_commit_bank_debug;
    wire [15:0] core_commit_version_debug;
    wire effective_commit_valid_debug;
    wire effective_commit_source_policy_debug;

    function automatic [15:0] crc16_42(input [41:0] data);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [47:0] padded;
        reg [7:0] octet;
        begin
            padded = {6'b0, data};
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 6; byte_index = byte_index + 1) begin
                octet = padded[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_42 = crc;
        end
    endfunction

    always @* begin
        input_payload = 42'd0;
        input_payload[9:0] = lfsr[9:0];
        input_payload[11:10] = (lfsr[13:12] == 2'd3) ? 2'd0 : lfsr[13:12];
        input_payload[13:12] = (lfsr[15:14] == 2'd3) ? 2'd0 : lfsr[15:14];
        input_payload[14] = lfsr[16];
        input_payload[22:15] = lfsr[24:17];
        input_payload[38:23] = {8'd0, lfsr[31:24]};
        input_payload[39] = lfsr[25] && (cycle_counter[6:0] == 7'd0);
        input_payload[40] = 1'b1;
        input_payload[41] = 1'b1;
    end
    assign in_word = {crc16_42(input_payload), input_payload};

    gkp_route_a_converged_production_top converged (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(1'b1),
        .in_word(in_word),
        .safe_boundary(safe_boundary),
        .cfg_begin_valid(cfg_begin_valid),
        .cfg_begin_bank(~active_bank_debug),
        .cfg_expected_active_version(active_version_debug),
        .cfg_new_image_version({8'd0, lfsr[31:24]} + 16'd2),
        .cfg_expected_crc32(lfsr ^ {16'd0, cycle_counter}),
        .cfg_word_valid(cfg_word_valid),
        .cfg_word_phase(lfsr[17]),
        .cfg_word_address(safe_address),
        .cfg_word_data(lfsr[21:0]),
        .cfg_finalize_valid(cfg_finalize_valid),
        .cfg_abort_valid(cfg_abort_valid),
        .host_commit_valid(host_commit_valid),
        .host_commit_bank(~active_bank_debug),
        .host_expected_active_version(active_version_debug),
        .host_new_activation_version(active_version_debug + 16'd1),
        .commit_cancel_valid(commit_cancel_valid),
        .management_snapshot_request(management_snapshot_request),
        .posterior_valid(cycle_counter[4:0] == 5'd0),
        .p_normal(p_normal),
        .p_smooth(p_smooth),
        .p_calibration(p_calibration),
        .p_burst(p_burst),
        .policy_ood_code(lfsr[31:24]),
        .router_boundary(cycle_counter[4:0] == 5'd31),
        .window_prequential_win(lfsr[22]),
        .integrity_fault(lfsr[23] && (cycle_counter[8:0] == 9'd0)),
        .version_fault(lfsr[24] && (cycle_counter[9:0] == 10'd0)),
        .integrity_clear(cycle_counter[7:0] == 8'hff),
        .leakage_event(lfsr[26] && (cycle_counter[7:0] == 8'd0)),
        .reset_ack(cycle_counter[7:0] == 8'h7f),
        .lkg_bank(lfsr[27]),
        .cfg_begin_ack(cfg_begin_ack),
        .cfg_word_ack(cfg_word_ack),
        .cfg_finalize_ack(cfg_finalize_ack),
        .cfg_abort_ack(cfg_abort_ack),
        .host_commit_ack(host_commit_ack),
        .policy_commit_ack(policy_commit_ack),
        .commit_complete(commit_complete),
        .commit_complete_source_policy(commit_complete_source_policy),
        .commit_cancel_ack(commit_cancel_ack),
        .management_snapshot_ack(management_snapshot_ack),
        .management_state_valid(management_state_valid),
        .management_reject(management_reject),
        .management_reject_reason(management_reject_reason),
        .management_ready(management_ready),
        .host_commit_blocked(host_commit_blocked),
        .out_word(out_word),
        .state_word(state_word),
        .route_action_word(route_action_word),
        .route_state_word(route_state_word),
        .route_version_word(route_version_word),
        .management_state_word(management_state_word),
        .map_valid_debug(map_valid_debug),
        .map_llr_debug(map_llr_debug),
        .map_address_debug(map_address_debug),
        .active_version_debug(active_version_debug),
        .active_bank_debug(active_bank_debug),
        .route_action_debug(route_action_debug),
        .route_reason_debug(route_reason_debug),
        .route_selected_bank_debug(route_selected_bank_debug),
        .route_commit_pending_debug(route_commit_pending_debug),
        .manager_commit_pending_debug(manager_commit_pending_debug),
        .manager_commit_pending_bank_debug(manager_commit_pending_bank_debug),
        .manager_commit_pending_version_debug(manager_commit_pending_version_debug),
        .manager_commit_pending_source_policy_debug(manager_commit_pending_source_policy_debug),
        .cfg_session_active_debug(cfg_session_active_debug),
        .cfg_staged_bank_debug(cfg_staged_bank_debug),
        .cfg_word_count_debug(cfg_word_count_debug),
        .cfg_all_words_received_debug(cfg_all_words_received_debug),
        .retired_bank_drain_count_debug(retired_bank_drain_count_debug),
        .bank0_trusted_debug(bank0_trusted_debug),
        .bank1_trusted_debug(bank1_trusted_debug),
        .bank0_image_version_debug(bank0_image_version_debug),
        .bank1_image_version_debug(bank1_image_version_debug),
        .core_cfg_we_debug(core_cfg_we_debug),
        .core_cfg_bank_debug(core_cfg_bank_debug),
        .core_commit_valid_debug(core_commit_valid_debug),
        .core_commit_bank_debug(core_commit_bank_debug),
        .core_commit_version_debug(core_commit_version_debug),
        .effective_commit_valid_debug(effective_commit_valid_debug),
        .effective_commit_source_policy_debug(effective_commit_source_policy_debug)
    );

    // Exact public-output concatenation width is 922 bits; pad to 29 words.
    wire [921:0] observable_payload = {
        effective_commit_source_policy_debug,
        effective_commit_valid_debug,
        core_commit_version_debug,
        core_commit_bank_debug,
        core_commit_valid_debug,
        core_cfg_bank_debug,
        core_cfg_we_debug,
        bank1_image_version_debug,
        bank0_image_version_debug,
        bank1_trusted_debug,
        bank0_trusted_debug,
        retired_bank_drain_count_debug,
        cfg_all_words_received_debug,
        cfg_word_count_debug,
        cfg_staged_bank_debug,
        cfg_session_active_debug,
        manager_commit_pending_source_policy_debug,
        manager_commit_pending_version_debug,
        manager_commit_pending_bank_debug,
        manager_commit_pending_debug,
        route_commit_pending_debug,
        route_selected_bank_debug,
        route_reason_debug,
        route_action_debug,
        active_bank_debug,
        active_version_debug,
        map_address_debug,
        map_llr_debug,
        map_valid_debug,
        management_state_word,
        route_version_word,
        route_state_word,
        route_action_word,
        state_word,
        out_word,
        host_commit_blocked,
        management_ready,
        management_reject_reason,
        management_reject,
        management_state_valid,
        management_snapshot_ack,
        commit_cancel_ack,
        commit_complete_source_policy,
        commit_complete,
        policy_commit_ack,
        host_commit_ack,
        cfg_abort_ack,
        cfg_finalize_ack,
        cfg_word_ack,
        cfg_begin_ack
    };
    wire [927:0] observable_bus = {6'd0, observable_payload};

    assign status_out = {
        signature[7][0], signature[6][0], signature[5][0], signature[4][0],
        signature[3][0], signature[2][0], signature[1][0], signature[0][0]
    };

    always @(posedge clk) begin
        if (!reset_n) begin
            lfsr <= {24'h510e52, seed_in ^ 8'h3c};
            cycle_counter <= 16'd0;
            for (index = 0; index < 29; index = index + 1)
                fold[index] <= 32'd0;
            for (index = 0; index < 8; index = index + 1)
                signature[index] <= 32'h9b05688c ^ index;
        end else begin
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            cycle_counter <= cycle_counter + 16'd1;
            for (index = 0; index < 29; index = index + 1)
                fold[index] <= observable_bus[index * 32 +: 32];
            for (index = 0; index < 8; index = index + 1) begin
                case (cycle_counter[1:0])
                    2'd0: signature[index] <= {signature[index][30:0], signature[index][31] ^ signature[index][21]} ^ fold[index];
                    2'd1: signature[index] <= {signature[index][30:0], signature[index][31] ^ signature[index][21]} ^ fold[index + 8];
                    2'd2: signature[index] <= {signature[index][30:0], signature[index][31] ^ signature[index][21]} ^ fold[index + 16];
                    default: signature[index] <= {signature[index][30:0], signature[index][31] ^ signature[index][21]} ^ ((index < 5) ? fold[index + 24] : 32'd0);
                endcase
            end
        end
    end
endmodule
