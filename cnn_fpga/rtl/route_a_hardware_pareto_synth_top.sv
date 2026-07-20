// T6.9.1 synthesis/P&R activity harness for the integrated Route-A fast path.
//
// The route_a_integrated_qualification_top remains the qualified functional
// top.  This wrapper only reduces the package pins and keeps all policy,
// event/fallback, A/B MAP-bank and optional student outputs observable during
// synthesis.  It is not a transport implementation or a board measurement.

module route_a_hardware_pareto_synth_top #(
    parameter integer ENABLE_STUDENT = 0
) (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] seed_in,
    output wire [7:0] status_out
);
    reg [31:0] lfsr;
    reg [15:0] cycle_counter;
    reg [31:0] signature;
    reg [31:0] fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7;
    reg [41:0] payload;

    wire [57:0] in_word;
    wire [117:0] out_word;
    wire [231:0] state_word;
    wire commit_ack;
    wire map_valid;
    wire [21:0] map_llr;
    wire [8:0] map_address;
    wire [15:0] active_version;
    wire active_bank;
    wire [2:0] route_action;
    wire [3:0] route_reason;
    wire route_selected_bank;
    wire route_commit_pending;
    wire host_commit_blocked;
    wire [79:0] route_action_word;
    wire [95:0] route_state_word;
    wire [63:0] route_version_word;

    wire cfg_activity = cycle_counter[11:0] == 12'hfff;
    wire [8:0] cfg_safe_address =
        (lfsr[8:0] <= 9'd256) ? lfsr[8:0] : (lfsr[8:0] - 9'd257);
    wire [5:0] posterior_a = lfsr[5:0];
    wire [5:0] posterior_b = lfsr[13:8];
    wire [5:0] posterior_c = lfsr[21:16];
    wire [7:0] p_normal = {2'b00, posterior_a};
    wire [7:0] p_smooth = {2'b00, posterior_b};
    wire [7:0] p_calibration = {2'b00, posterior_c};
    wire [7:0] p_burst = 8'd255 - p_normal - p_smooth - p_calibration;

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
        payload = 42'd0;
        payload[9:0] = lfsr[9:0];
        payload[11:10] = (lfsr[13:12] == 2'd3) ? 2'd0 : lfsr[13:12];
        payload[13:12] = (lfsr[15:14] == 2'd3) ? 2'd0 : lfsr[15:14];
        payload[14] = lfsr[16];
        payload[22:15] = lfsr[24:17] & 8'h7f;
        payload[38:23] = {8'd0, lfsr[31:24]};
        payload[39] = lfsr[25] && (cycle_counter[5:0] == 6'd0);
        payload[40] = 1'b1;
        payload[41] = 1'b1;
    end
    assign in_word = {crc16_42(payload), payload};

    route_a_integrated_qualification_top integrated (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(1'b1),
        .in_word(in_word),
        .safe_boundary(1'b1),
        .host_commit_valid(cycle_counter[12:0] == 13'd4000),
        .host_commit_bank(~active_bank),
        .host_commit_version(active_version + 16'd1),
        .cfg_we(cfg_activity),
        .cfg_bank(~active_bank),
        .cfg_phase(lfsr[17]),
        .cfg_address(cfg_safe_address),
        .cfg_data(lfsr[21:0]),
        .bank0_trusted(1'b1),
        .bank1_trusted(1'b1),
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
        .lkg_bank(1'b0),
        .commit_ack(commit_ack),
        .out_word(out_word),
        .state_word(state_word),
        .map_valid_debug(map_valid),
        .map_llr_debug(map_llr),
        .map_address_debug(map_address),
        .active_version_debug(active_version),
        .active_bank_debug(active_bank),
        .route_action_debug(route_action),
        .route_reason_debug(route_reason),
        .route_selected_bank_debug(route_selected_bank),
        .route_commit_pending_debug(route_commit_pending),
        .host_commit_blocked_debug(host_commit_blocked),
        .route_action_word(route_action_word),
        .route_state_word(route_state_word),
        .route_version_word(route_version_word)
    );

    wire student_busy;
    wire student_done;
    wire student_out_valid;
    wire [3:0] student_out_index;
    wire signed [17:0] student_out_value;
    wire [71:0] student_state;

    generate
        if (ENABLE_STUDENT != 0) begin : student_sidecar
            wire student_start = !student_busy && (cycle_counter[6:0] == 7'd0);
            low_dimensional_student_kernel student (
                .clk(clk),
                .reset_n(reset_n),
                .start(student_start),
                .outcome_e(payload[10]),
                .health_ok(route_action == 3'd0),
                .busy(student_busy),
                .done(student_done),
                .out_valid(student_out_valid),
                .out_index(student_out_index),
                .out_value(student_out_value),
                .state_debug(student_state)
            );
        end else begin : no_student_sidecar
            assign student_busy = 1'b0;
            assign student_done = 1'b0;
            assign student_out_valid = 1'b0;
            assign student_out_index = 4'd0;
            assign student_out_value = 18'sd0;
            assign student_state = 72'd0;
        end
    endgenerate

    assign status_out = signature[7:0] ^
        {route_action, route_reason[1:0], active_bank, commit_ack, map_valid};

    always @(posedge clk) begin
        if (!reset_n) begin
            lfsr <= {24'h6a09e6, seed_in ^ 8'ha5};
            cycle_counter <= 16'd0;
            signature <= 32'hbb67ae85;
            fold0 <= 32'd0;
            fold1 <= 32'd0;
            fold2 <= 32'd0;
            fold3 <= 32'd0;
            fold4 <= 32'd0;
            fold5 <= 32'd0;
            fold6 <= 32'd0;
            fold7 <= 32'd0;
        end else begin
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            cycle_counter <= cycle_counter + 16'd1;
            fold0 <= out_word[31:0] ^ out_word[63:32];
            fold1 <= out_word[95:64] ^ {10'd0, out_word[117:96]};
            fold2 <= state_word[31:0] ^ state_word[95:64] ^ state_word[159:128];
            fold3 <= state_word[63:32] ^ state_word[127:96] ^ state_word[191:160];
            fold4 <= state_word[223:192] ^ {24'd0, state_word[231:224]};
            fold5 <= route_action_word[31:0] ^ route_action_word[63:32] ^
                     {16'd0, route_action_word[79:64]};
            fold6 <= route_state_word[31:0] ^ route_state_word[63:32] ^
                     route_state_word[95:64] ^ route_version_word[31:0] ^
                     route_version_word[63:32];
            fold7 <= {student_state[7:0], student_out_value[8:0], student_out_index,
                      student_out_valid, student_done, student_busy,
                      route_selected_bank, route_commit_pending,
                      host_commit_blocked, map_address[2:0]};
            case (cycle_counter[2:0])
                3'd0: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold0;
                3'd1: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold1;
                3'd2: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold2;
                3'd3: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold3;
                3'd4: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold4;
                3'd5: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold5;
                3'd6: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold6;
                default: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold7;
            endcase
        end
    end
endmodule
