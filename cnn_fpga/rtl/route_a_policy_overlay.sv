// T6.7.3: synthesizable integer Route-A policy and trusted-bank router.
//
// Posterior order: normal, smooth, calibration_shift, burst.  The slow loop
// owns posterior inference; this fast-path overlay owns only the quantized
// action contract, hysteresis, safety latches and A/B expert selection.

module route_a_policy_overlay (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          sample_valid,
    input  wire          safe_boundary,
    input  wire          posterior_valid,
    input  wire [7:0]    p_normal,
    input  wire [7:0]    p_smooth,
    input  wire [7:0]    p_calibration,
    input  wire [7:0]    p_burst,
    input  wire [7:0]    ood_code,
    input  wire          router_boundary,
    input  wire          window_prequential_win,
    input  wire          integrity_fault,
    input  wire          version_fault,
    input  wire          integrity_clear,
    input  wire          leakage_event,
    input  wire          reset_ack,
    input  wire          lkg_bank,
    input  wire          active_bank,
    input  wire [15:0]   active_version,
    input  wire [117:0]  core_out_word,

    output wire          auto_commit_valid,
    output wire          auto_commit_bank,
    output wire [15:0]   auto_commit_version,
    output wire [2:0]    action_debug,
    output wire [3:0]    reason_debug,
    output wire          selected_bank_debug,
    output wire          commit_pending_debug,
    output wire [79:0]   route_action_word,
    output wire [95:0]   route_state_word,
    output wire [63:0]   route_version_word
);

    localparam [2:0] ACTION_OPEN               = 3'd0;
    localparam [2:0] ACTION_TAIL_EWMA          = 3'd1;
    localparam [2:0] ACTION_UNCERTAIN_EWMA     = 3'd2;
    localparam [2:0] ACTION_LEAKAGE_RESET      = 3'd3;
    localparam [2:0] ACTION_INTEGRITY_ROLLBACK = 3'd4;

    localparam [3:0] REASON_ADAPTIVE_READY      = 4'd0;
    localparam [3:0] REASON_RAW_TAIL            = 4'd1;
    localparam [3:0] REASON_OOD_EVENT           = 4'd2;
    localparam [3:0] REASON_TAIL_LATCHED        = 4'd3;
    localparam [3:0] REASON_POSTERIOR_UNCERTAIN = 4'd4;
    localparam [3:0] REASON_LEAKAGE             = 4'd5;
    localparam [3:0] REASON_INTEGRITY           = 4'd6;
    localparam [3:0] REASON_POSTERIOR_SUM       = 4'd7;
    localparam [3:0] REASON_VERSION             = 4'd8;

    localparam [8:0] TAIL_ENTER_CODE = 9'd230;
    localparam [8:0] TAIL_EXIT_CODE  = 9'd51;

    function automatic [15:0] crc16_48(input [47:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [7:0] octet;
        begin
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 6; byte_index = byte_index + 1) begin
                octet = payload[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_48 = crc;
        end
    endfunction

    function automatic [15:0] crc16_64(input [63:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [7:0] octet;
        begin
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 8; byte_index = byte_index + 1) begin
                octet = payload[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_64 = crc;
        end
    endfunction

    function automatic [15:0] crc16_80(input [79:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [7:0] octet;
        begin
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 10; byte_index = byte_index + 1) begin
                octet = payload[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_80 = crc;
        end
    endfunction

    reg [2:0] action;
    reg [3:0] reason;
    reg tail_latched;
    reg leakage_latched;
    reg integrity_latched;
    reg [1:0] enter_run;
    reg [3:0] recovery_run;
    reg selected_bank;
    reg commit_pending;
    reg pending_bank;
    reg [15:0] policy_update_count;
    reg [15:0] fallback_count;
    reg [7:0] rollback_count;
    reg [15:0] last_source_version;

    reg [2:0] pipe_action [0:6];
    reg [3:0] pipe_reason [0:6];
    reg pipe_selected_bank [0:6];
    reg pipe_active_bank [0:6];
    reg [15:0] pipe_version [0:6];
    reg pipe_valid [0:6];

    wire [9:0] posterior_sum = {2'b0, p_normal} + {2'b0, p_smooth} +
                                {2'b0, p_calibration} + {2'b0, p_burst};
    wire posterior_sum_fault = posterior_valid && (posterior_sum != 10'd255);
    wire [8:0] tail_code = {1'b0, p_calibration} + {1'b0, p_burst};
    wire [8:0] adaptive_code = {1'b0, p_normal} + {1'b0, p_smooth};
    wire [7:0] max01 = (p_normal > p_smooth) ? p_normal : p_smooth;
    wire [7:0] max23 = (p_calibration > p_burst) ? p_calibration : p_burst;
    wire [7:0] max_posterior = (max01 > max23) ? max01 : max23;
    wire [7:0] uncertainty_code = 8'd255 - max_posterior;
    wire raw_tail = tail_code >= TAIL_ENTER_CODE;
    wire event_alert = ood_code > 8'd192;
    wire adaptive_ready = (adaptive_code >= TAIL_ENTER_CODE) &&
                          (uncertainty_code < 8'd64);
    wire integrity_now = integrity_fault || version_fault || posterior_sum_fault;

    reg [2:0] action_next;
    reg [3:0] reason_next;
    reg tail_latched_next;
    reg leakage_latched_next;
    reg integrity_latched_next;
    reg [1:0] enter_run_next;
    reg [3:0] recovery_run_next;
    reg selected_bank_next;

    always @* begin
        action_next = action;
        reason_next = reason;
        tail_latched_next = tail_latched;
        leakage_latched_next = leakage_latched || leakage_event;
        integrity_latched_next = integrity_latched || integrity_now;
        enter_run_next = enter_run;
        recovery_run_next = recovery_run;

        if (integrity_latched_next && integrity_clear && !integrity_now &&
            (active_bank == lkg_bank))
            integrity_latched_next = 1'b0;
        if (leakage_latched_next && reset_ack && !leakage_event)
            leakage_latched_next = 1'b0;

        if (posterior_valid) begin
            enter_run_next = (raw_tail || event_alert) ?
                             ((enter_run == 2'd3) ? 2'd3 : enter_run + 2'd1) : 2'd0;
            recovery_run_next = ((tail_code <= TAIL_EXIT_CODE) && adaptive_ready) ?
                                ((recovery_run == 4'd15) ? 4'd15 : recovery_run + 4'd1) : 4'd0;
            if (enter_run_next >= 2'd2)
                tail_latched_next = 1'b1;
            else if (recovery_run_next >= 4'd8)
                tail_latched_next = 1'b0;
        end

        if (integrity_latched_next) begin
            action_next = ACTION_INTEGRITY_ROLLBACK;
            if (version_fault)
                reason_next = REASON_VERSION;
            else if (posterior_sum_fault)
                reason_next = REASON_POSTERIOR_SUM;
            else
                reason_next = REASON_INTEGRITY;
        end else if (leakage_latched_next) begin
            action_next = ACTION_LEAKAGE_RESET;
            reason_next = REASON_LEAKAGE;
        end else if (posterior_valid) begin
            if (raw_tail) begin
                action_next = ACTION_TAIL_EWMA;
                reason_next = REASON_RAW_TAIL;
            end else if (event_alert) begin
                action_next = ACTION_TAIL_EWMA;
                reason_next = REASON_OOD_EVENT;
            end else if (tail_latched_next) begin
                action_next = ACTION_TAIL_EWMA;
                reason_next = REASON_TAIL_LATCHED;
            end else if (!adaptive_ready) begin
                action_next = ACTION_UNCERTAIN_EWMA;
                reason_next = REASON_POSTERIOR_UNCERTAIN;
            end else begin
                action_next = ACTION_OPEN;
                reason_next = REASON_ADAPTIVE_READY;
            end
        end

        selected_bank_next = selected_bank;
        if (action_next == ACTION_INTEGRITY_ROLLBACK)
            selected_bank_next = lkg_bank;
        else if ((action_next == ACTION_TAIL_EWMA) ||
                 (action_next == ACTION_UNCERTAIN_EWMA) ||
                 (action_next == ACTION_LEAKAGE_RESET))
            selected_bank_next = 1'b0;
        else if (router_boundary)
            selected_bank_next = window_prequential_win && (p_smooth >= 8'd77);
    end

    assign auto_commit_valid = commit_pending && safe_boundary &&
                               (pending_bank != active_bank) &&
                               (active_version != 16'hffff);
    assign auto_commit_bank = pending_bank;
    assign auto_commit_version = active_version + 16'd1;
    assign action_debug = action;
    assign reason_debug = reason;
    assign selected_bank_debug = selected_bank;
    assign commit_pending_debug = commit_pending;

    integer index;
    always @(posedge clk) begin
        if (!reset_n) begin
            action <= ACTION_OPEN;
            reason <= REASON_ADAPTIVE_READY;
            tail_latched <= 1'b0;
            leakage_latched <= 1'b0;
            integrity_latched <= 1'b0;
            enter_run <= 2'd0;
            recovery_run <= 4'd0;
            selected_bank <= 1'b0;
            commit_pending <= 1'b0;
            pending_bank <= 1'b0;
            policy_update_count <= 16'd0;
            fallback_count <= 16'd0;
            rollback_count <= 8'd0;
            last_source_version <= 16'd0;
            for (index = 0; index < 7; index = index + 1) begin
                pipe_valid[index] <= 1'b0;
                pipe_action[index] <= ACTION_OPEN;
                pipe_reason[index] <= REASON_ADAPTIVE_READY;
                pipe_selected_bank[index] <= 1'b0;
                pipe_active_bank[index] <= 1'b0;
                pipe_version[index] <= 16'd0;
            end
        end else begin
            action <= action_next;
            reason <= reason_next;
            tail_latched <= tail_latched_next;
            leakage_latched <= leakage_latched_next;
            integrity_latched <= integrity_latched_next;
            enter_run <= enter_run_next;
            recovery_run <= recovery_run_next;
            selected_bank <= selected_bank_next;

            if (commit_pending && (active_bank == pending_bank))
                commit_pending <= 1'b0;
            if (selected_bank_next != active_bank) begin
                commit_pending <= 1'b1;
                pending_bank <= selected_bank_next;
            end else if (pending_bank == active_bank) begin
                commit_pending <= 1'b0;
            end

            if (posterior_valid && (policy_update_count != 16'hffff))
                policy_update_count <= policy_update_count + 16'd1;
            if ((action_next != ACTION_OPEN) && (fallback_count != 16'hffff))
                fallback_count <= fallback_count + 16'd1;
            if ((action_next == ACTION_INTEGRITY_ROLLBACK) &&
                (action != ACTION_INTEGRITY_ROLLBACK) && (rollback_count != 8'hff))
                rollback_count <= rollback_count + 8'd1;

            pipe_valid[0] <= sample_valid;
            pipe_action[0] <= action_next;
            pipe_reason[0] <= reason_next;
            pipe_selected_bank[0] <= selected_bank_next;
            pipe_active_bank[0] <= active_bank;
            pipe_version[0] <= active_version;
            for (index = 1; index < 7; index = index + 1) begin
                pipe_valid[index] <= pipe_valid[index - 1];
                pipe_action[index] <= pipe_action[index - 1];
                pipe_reason[index] <= pipe_reason[index - 1];
                pipe_selected_bank[index] <= pipe_selected_bank[index - 1];
                pipe_active_bank[index] <= pipe_active_bank[index - 1];
                pipe_version[index] <= pipe_version[index - 1];
            end
            // At the sampling edge the core's newly visible out_word aligns
            // with the pre-edge stage 5 metadata.  Stage 6 is used by the
            // combinational action word after that same edge.
            if (pipe_valid[5])
                last_source_version <= pipe_version[5];
        end
    end

    reg [63:0] action_payload;
    always @* begin
        action_payload = 64'd0;
        if (pipe_valid[6] && core_out_word[0]) begin
            action_payload[0] = 1'b1;
            action_payload[3:1] = pipe_action[6];
            action_payload[7:4] = pipe_reason[6];
            action_payload[8] = pipe_selected_bank[6];
            action_payload[9] = pipe_action[6] != ACTION_OPEN;
            action_payload[10] = core_out_word[4];
            action_payload[11] = core_out_word[5];
            action_payload[13:12] = core_out_word[8:7];
            action_payload[16:14] = core_out_word[3:1];
            action_payload[19:17] = core_out_word[49:47];
            action_payload[33:20] = core_out_word[63:50];
            action_payload[49:34] = pipe_version[6];
            action_payload[50] = pipe_active_bank[6];
            action_payload[63:51] = policy_update_count[12:0];
        end
    end
    assign route_action_word = {crc16_64(action_payload), action_payload};

    wire [79:0] state_payload = {
        4'd0,
        rollback_count,
        fallback_count,
        policy_update_count,
        active_version,
        active_bank,
        pending_bank,
        commit_pending,
        selected_bank,
        recovery_run,
        enter_run,
        integrity_latched,
        leakage_latched,
        tail_latched,
        reason,
        action
    };
    assign route_state_word = {crc16_80(state_payload), state_payload};

    wire [47:0] version_payload = {
        policy_update_count[11:0],
        commit_pending,
        pending_bank,
        active_bank,
        selected_bank,
        last_source_version,
        active_version
    };
    assign route_version_word = {crc16_48(version_payload), version_payload};

endmodule
