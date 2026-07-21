// T6.25.2 proof harness for the actual six-cycle gkp_fast_path_core.
// Two valid samples are injected at II=1: the first misses its deadline and
// the second exceeds the parameter-age ceiling.  Both must emerge on adjacent
// cycles as explicit fallback decisions with no frame/action mutation.

module gkp_fast_path_fail_closed_formal (
    input wire clk,
    output wire prop_fail_closed_outputs,
    output reg cover_two_adjacent_fault_outputs
);
    localparam [2:0] MODE_FALLBACK = 3'd5;

    (* anyseq *) reg reset_n;
    reg f_past_valid;
    reg [1:0] issue_count;
    reg [6:0] deadline_pipe;
    reg [6:0] age_pipe;

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

    function automatic [15:0] crc16_42(input [41:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [47:0] padded;
        reg [7:0] octet;
        begin
            padded = {6'b0, payload};
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

    wire in_valid = reset_n && (issue_count < 2'd2);
    reg [41:0] input_payload;
    always @* begin
        input_payload = 42'd0;
        input_payload[40] = 1'b1; // observation_valid
        if (issue_count == 2'd0) begin
            input_payload[41] = 1'b0; // explicit deadline violation
            input_payload[38:23] = 16'd0;
        end else begin
            input_payload[41] = 1'b1;
            input_payload[38:23] = 16'd65; // MAX_PARAMETER_AGE_CYCLES + 1
        end
    end
    wire [57:0] in_word = {crc16_42(input_payload), input_payload};

    wire [117:0] out_word;
    wire [231:0] state_word;
    wire commit_ack;
    wire map_valid_debug;
    wire [21:0] map_llr_debug;
    wire [8:0] map_address_debug;
    wire [15:0] active_version_debug;
    wire active_bank_debug;

    gkp_fast_path_core #(
        .OOD_THRESHOLD_CODE(8'd192),
        .MAX_PARAMETER_AGE_CYCLES(16'd64),
        .MAX_TRUSTED_BANK_VERSION(16'hffff)
    ) core (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(in_valid),
        .in_word(in_word),
        .safe_boundary(1'b0),
        .commit_valid(1'b0),
        .commit_bank(1'b0),
        .commit_version(16'd0),
        .commit_ack(commit_ack),
        .cfg_we(1'b0),
        .cfg_bank(1'b0),
        .cfg_phase(1'b0),
        .cfg_address(9'd0),
        .cfg_data(22'd0),
        .bank0_trusted(1'b1),
        .bank1_trusted(1'b1),
        .out_word(out_word),
        .state_word(state_word),
        .map_valid_debug(map_valid_debug),
        .map_llr_debug(map_llr_debug),
        .map_address_debug(map_address_debug),
        .active_version_debug(active_version_debug),
        .active_bank_debug(active_bank_debug)
    );

    wire output_due = deadline_pipe[6] || age_pipe[6];
    wire common_fail_closed = !output_due ||
        (out_word[0] && out_word[3:1] == MODE_FALLBACK &&
         !out_word[4] && !out_word[6] && out_word[8:7] == 2'd0 &&
         !out_word[9] && !out_word[10]);
    // event_fault_mask[9:8] is packed into output_payload[59:58].
    wire deadline_fault_visible = !deadline_pipe[6] || out_word[59];
    wire age_fault_visible = !age_pipe[6] || out_word[58];
    assign prop_fail_closed_outputs = common_fail_closed &&
                                      deadline_fault_visible &&
                                      age_fault_visible;

    always @* begin
        if (reset_n)
            assert(prop_fail_closed_outputs);
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            issue_count <= 2'd0;
            deadline_pipe <= 7'd0;
            age_pipe <= 7'd0;
            cover_two_adjacent_fault_outputs <= 1'b0;
        end else begin
            if (in_valid)
                issue_count <= issue_count + 2'd1;
            deadline_pipe <= {deadline_pipe[5:0], in_valid && issue_count == 2'd0};
            age_pipe <= {age_pipe[5:0], in_valid && issue_count == 2'd1};
            cover_two_adjacent_fault_outputs <= cover_two_adjacent_fault_outputs ||
                (deadline_pipe[6] && age_pipe[5] && prop_fail_closed_outputs);
        end
    end
endmodule
