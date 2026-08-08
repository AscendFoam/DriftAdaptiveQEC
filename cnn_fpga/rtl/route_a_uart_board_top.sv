// Tang Nano 20K candidate board top for Route-A framed replay.
//
// This top is intentionally a pre-board artifact.  It uses the official
// 27 MHz clock and UART pins, but physical BL616/UART behavior, board timing,
// and source-to-action latency remain unqualified until T6.1/T6.2/T6.4.
//
// Request: 40 fixed bytes, response: 96 fixed bytes, little-endian fields.
// CRC is reflected CRC-32/IEEE over every byte except the final four CRC bytes.

module route_a_uart_board_top #(
    parameter integer CLK_HZ = 27000000,
    parameter integer UART_BAUD = 3000000,
    parameter integer FRAME_TIMEOUT_CYCLES = 270000
) (
    input  wire clk,
    input  wire reset_n,
    input  wire uart_rx,
    output wire uart_tx,
    output reg  action_strobe,
    output reg  error_strobe
);
    localparam [7:0] PROTOCOL_VERSION = 8'd1;
    localparam [7:0] CMD_EXECUTE = 8'h01;
    localparam [7:0] CMD_STATUS = 8'h02;
    localparam [7:0] CMD_CLEAR_COUNTERS = 8'h03;
    localparam [7:0] STATUS_OK = 8'h00;
    localparam [7:0] STATUS_DUPLICATE_REPLAY = 8'h01;
    localparam [7:0] STATUS_BAD_CRC = 8'h10;
    localparam [7:0] STATUS_BAD_VERSION = 8'h11;
    localparam [7:0] STATUS_BAD_LENGTH = 8'h12;
    localparam [7:0] STATUS_BAD_RESERVED = 8'h13;
    localparam [7:0] STATUS_BAD_SEQUENCE = 8'h14;
    localparam [7:0] STATUS_BAD_COMMAND = 8'h15;

    localparam [3:0] C_IDLE = 4'd0;
    localparam [3:0] C_DISPATCH = 4'd1;
    localparam [3:0] C_EXEC_WAIT = 4'd2;
    localparam [3:0] C_CLEAR = 4'd3;
    localparam [3:0] C_BUILD = 4'd4;
    localparam [3:0] C_CRC = 4'd5;
    localparam [3:0] C_READY = 4'd6;
    localparam [3:0] C_TX = 4'd7;

    function automatic [31:0] crc32_byte(input [31:0] crc_in, input [7:0] data);
        integer bit_index;
        reg [31:0] crc;
        begin
            crc = crc_in ^ data;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                crc = crc[0] ? ((crc >> 1) ^ 32'hedb88320) : (crc >> 1);
            crc32_byte = crc;
        end
    endfunction

    wire [7:0] rx_byte;
    wire rx_byte_valid;
    wire rx_framing_error;
    wire rx_busy;
    reg [7:0] tx_byte;
    reg tx_byte_valid;
    wire tx_byte_ready;
    wire tx_busy;

    route_a_uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(UART_BAUD)) uart_receiver (
        .clk(clk), .reset_n(reset_n), .rx_pin(uart_rx),
        .byte_data(rx_byte), .byte_valid(rx_byte_valid),
        .framing_error(rx_framing_error), .busy(rx_busy)
    );
    route_a_uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(UART_BAUD)) uart_transmitter (
        .clk(clk), .reset_n(reset_n), .byte_data(tx_byte),
        .byte_valid(tx_byte_valid), .byte_ready(tx_byte_ready),
        .tx_pin(uart_tx), .busy(tx_busy)
    );

    reg [7:0] rx_frame [0:39];
    reg [5:0] rx_index;
    reg [31:0] request_crc;
    reg [31:0] frame_timeout;
    reg [191:0] request_payload;
    reg [31:0] request_sequence;
    reg [7:0] request_command;
    reg [7:0] response_status;
    reg [31:0] expected_sequence;
    reg [31:0] last_sequence;
    reg last_response_valid;

    reg [767:0] response_frame;
    reg [6:0] response_crc_index;
    reg [31:0] response_crc;
    wire [31:0] response_crc_next = crc32_byte(
        response_crc, response_frame[response_crc_index * 8 +: 8]
    );
    reg [6:0] tx_index;
    reg [3:0] control_state;

    reg [15:0] rx_crc_errors;
    reg [15:0] sequence_errors;
    reg [15:0] uart_framing_errors;
    reg [15:0] frame_timeout_errors;
    reg [15:0] dropped_rx_bytes;
    reg [63:0] cycle_counter;
    reg [63:0] response_start_cycle;
    reg [63:0] response_action_cycle;

    wire [57:0] req_in_word = request_payload[57:0];
    wire req_safe_boundary = request_payload[58];
    wire req_host_commit_valid = request_payload[59];
    wire req_host_commit_bank = request_payload[60];
    wire [15:0] req_host_commit_version = request_payload[76:61];
    wire req_cfg_we = request_payload[77];
    wire req_cfg_bank = request_payload[78];
    wire req_cfg_phase = request_payload[79];
    wire [8:0] req_cfg_address = request_payload[88:80];
    wire [21:0] req_cfg_data = request_payload[110:89];
    wire req_bank0_trusted = request_payload[111];
    wire req_bank1_trusted = request_payload[112];
    wire req_posterior_valid = request_payload[113];
    wire [7:0] req_p_normal = request_payload[121:114];
    wire [7:0] req_p_smooth = request_payload[129:122];
    wire [7:0] req_p_calibration = request_payload[137:130];
    wire [7:0] req_p_burst = request_payload[145:138];
    wire [7:0] req_policy_ood_code = request_payload[153:146];
    wire req_router_boundary = request_payload[154];
    wire req_window_prequential_win = request_payload[155];
    wire req_integrity_fault = request_payload[156];
    wire req_version_fault = request_payload[157];
    wire req_integrity_clear = request_payload[158];
    wire req_leakage_event = request_payload[159];
    wire req_reset_ack = request_payload[160];
    wire req_lkg_bank = request_payload[161];

    reg core_in_valid;
    wire commit_ack;
    wire [117:0] out_word;
    wire [231:0] state_word;
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

    route_a_integrated_qualification_top integrated (
        .clk(clk), .reset_n(reset_n), .in_valid(core_in_valid),
        .in_word(req_in_word), .safe_boundary(req_safe_boundary),
        // Event-like host controls are sampled exactly once with the execute
        // transaction.  Leaving these request bits level-high while the UART
        // is idle would otherwise repeat writes, commits, or fault events.
        .host_commit_valid(req_host_commit_valid && core_in_valid),
        .host_commit_bank(req_host_commit_bank),
        .host_commit_version(req_host_commit_version),
        .cfg_we(req_cfg_we && core_in_valid), .cfg_bank(req_cfg_bank), .cfg_phase(req_cfg_phase),
        .cfg_address(req_cfg_address), .cfg_data(req_cfg_data),
        .bank0_trusted(req_bank0_trusted), .bank1_trusted(req_bank1_trusted),
        .posterior_valid(req_posterior_valid && core_in_valid), .p_normal(req_p_normal),
        .p_smooth(req_p_smooth), .p_calibration(req_p_calibration),
        .p_burst(req_p_burst), .policy_ood_code(req_policy_ood_code),
        .router_boundary(req_router_boundary && core_in_valid),
        .window_prequential_win(req_window_prequential_win),
        .integrity_fault(req_integrity_fault && core_in_valid),
        .version_fault(req_version_fault && core_in_valid),
        .integrity_clear(req_integrity_clear && core_in_valid),
        .leakage_event(req_leakage_event && core_in_valid),
        .reset_ack(req_reset_ack && core_in_valid), .lkg_bank(req_lkg_bank),
        .commit_ack(commit_ack), .out_word(out_word), .state_word(state_word),
        .map_valid_debug(map_valid), .map_llr_debug(map_llr),
        .map_address_debug(map_address), .active_version_debug(active_version),
        .active_bank_debug(active_bank), .route_action_debug(route_action),
        .route_reason_debug(route_reason),
        .route_selected_bank_debug(route_selected_bank),
        .route_commit_pending_debug(route_commit_pending),
        .host_commit_blocked_debug(host_commit_blocked),
        .route_action_word(route_action_word), .route_state_word(route_state_word),
        .route_version_word(route_version_word)
    );

    reg [117:0] capture_out_word;
    reg [79:0] capture_route_action_word;
    reg [95:0] capture_route_state_word;
    reg [63:0] capture_route_version_word;
    reg [21:0] capture_map_llr;
    reg [8:0] capture_map_address;
    reg [15:0] capture_active_version;
    reg [7:0] capture_debug_flags;

    integer byte_index;
    reg [31:0] received_request_crc;
    reg [31:0] received_sequence;
    wire parser_busy = rx_index != 0;
    wire transport_busy = control_state != C_IDLE || tx_busy || tx_byte_valid;

    always @(posedge clk) begin
        if (!reset_n) begin
            rx_index <= 6'd0;
            request_crc <= 32'hffffffff;
            frame_timeout <= 32'd0;
            request_payload <= 192'd0;
            request_sequence <= 32'd0;
            request_command <= 8'd0;
            response_status <= STATUS_OK;
            expected_sequence <= 32'd0;
            last_sequence <= 32'd0;
            last_response_valid <= 1'b0;
            response_frame <= 768'd0;
            response_crc_index <= 7'd0;
            response_crc <= 32'hffffffff;
            tx_index <= 7'd0;
            tx_byte <= 8'd0;
            tx_byte_valid <= 1'b0;
            control_state <= C_IDLE;
            rx_crc_errors <= 16'd0;
            sequence_errors <= 16'd0;
            uart_framing_errors <= 16'd0;
            frame_timeout_errors <= 16'd0;
            dropped_rx_bytes <= 16'd0;
            cycle_counter <= 64'd0;
            response_start_cycle <= 64'd0;
            response_action_cycle <= 64'd0;
            core_in_valid <= 1'b0;
            capture_out_word <= 118'd0;
            capture_route_action_word <= 80'd0;
            capture_route_state_word <= 96'd0;
            capture_route_version_word <= 64'd0;
            capture_map_llr <= 22'd0;
            capture_map_address <= 9'd0;
            capture_active_version <= 16'd0;
            capture_debug_flags <= 8'd0;
            action_strobe <= 1'b0;
            error_strobe <= 1'b0;
            received_request_crc <= 32'd0;
            received_sequence <= 32'd0;
            for (byte_index = 0; byte_index < 40; byte_index = byte_index + 1)
                rx_frame[byte_index] <= 8'd0;
        end else begin
            cycle_counter <= cycle_counter + 64'd1;
            action_strobe <= 1'b0;
            error_strobe <= 1'b0;
            core_in_valid <= 1'b0;

            if (tx_byte_valid && tx_byte_ready) begin
                tx_byte_valid <= 1'b0;
                if (tx_index == 7'd95) begin
                    tx_index <= 7'd0;
                    control_state <= C_IDLE;
                end else begin
                    tx_index <= tx_index + 7'd1;
                end
            end else if ((control_state == C_TX) && !tx_byte_valid && tx_byte_ready) begin
                tx_byte <= response_frame[tx_index * 8 +: 8];
                tx_byte_valid <= 1'b1;
            end

            if (rx_framing_error) begin
                if (uart_framing_errors != 16'hffff)
                    uart_framing_errors <= uart_framing_errors + 16'd1;
                rx_index <= 6'd0;
                frame_timeout <= 32'd0;
                error_strobe <= 1'b1;
            end else if (parser_busy && !rx_byte_valid) begin
                if (frame_timeout >= FRAME_TIMEOUT_CYCLES - 1) begin
                    rx_index <= 6'd0;
                    frame_timeout <= 32'd0;
                    if (frame_timeout_errors != 16'hffff)
                        frame_timeout_errors <= frame_timeout_errors + 16'd1;
                    error_strobe <= 1'b1;
                end else begin
                    frame_timeout <= frame_timeout + 32'd1;
                end
            end

            if (rx_byte_valid) begin
                frame_timeout <= 32'd0;
                if (transport_busy) begin
                    if (dropped_rx_bytes != 16'hffff)
                        dropped_rx_bytes <= dropped_rx_bytes + 16'd1;
                    rx_index <= 6'd0;
                    error_strobe <= 1'b1;
                end else if (rx_index == 6'd0) begin
                    if (rx_byte == 8'ha5) begin
                        rx_frame[0] <= rx_byte;
                        request_crc <= crc32_byte(32'hffffffff, rx_byte);
                        rx_index <= 6'd1;
                    end
                end else if (rx_index == 6'd1) begin
                    if (rx_byte == 8'h5a) begin
                        rx_frame[1] <= rx_byte;
                        request_crc <= crc32_byte(request_crc, rx_byte);
                        rx_index <= 6'd2;
                    end else if (rx_byte == 8'ha5) begin
                        rx_frame[0] <= rx_byte;
                        request_crc <= crc32_byte(32'hffffffff, rx_byte);
                        rx_index <= 6'd1;
                    end else begin
                        rx_index <= 6'd0;
                    end
                end else begin
                    rx_frame[rx_index] <= rx_byte;
                    if (rx_index <= 6'd35)
                        request_crc <= crc32_byte(request_crc, rx_byte);
                    if (rx_index == 6'd39) begin
                        rx_index <= 6'd0;
                        received_request_crc = {rx_byte, rx_frame[38], rx_frame[37], rx_frame[36]};
                        received_sequence = {rx_frame[7], rx_frame[6], rx_frame[5], rx_frame[4]};
                        request_sequence <= received_sequence;
                        request_command <= rx_frame[3];
                        response_start_cycle <= cycle_counter;
                        response_action_cycle <= cycle_counter;
                        capture_out_word <= 118'd0;
                        capture_route_action_word <= 80'd0;
                        capture_route_state_word <= 96'd0;
                        capture_route_version_word <= 64'd0;
                        capture_map_llr <= 22'd0;
                        capture_map_address <= 9'd0;
                        capture_active_version <= active_version;
                        capture_debug_flags <= 8'd0;
                        if (received_request_crc != ~request_crc) begin
                            response_status <= STATUS_BAD_CRC;
                            if (rx_crc_errors != 16'hffff)
                                rx_crc_errors <= rx_crc_errors + 16'd1;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else if (rx_frame[2] != PROTOCOL_VERSION) begin
                            response_status <= STATUS_BAD_VERSION;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else if (rx_frame[8] != 8'd24) begin
                            response_status <= STATUS_BAD_LENGTH;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else if ((rx_frame[9] != 0) || (rx_frame[10] != 0) ||
                                     (rx_frame[11] != 0) ||
                                     ((rx_frame[32] & 8'hfc) != 0) ||
                                     (rx_frame[33] != 0) || (rx_frame[34] != 0) ||
                                     (rx_frame[35] != 0)) begin
                            response_status <= STATUS_BAD_RESERVED;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else if (last_response_valid && (received_sequence == last_sequence)) begin
                            // Duplicate frames are idempotent but are not
                            // re-executed and do not consume a sequence.  A
                            // compact explicit response avoids a 1024-bit
                            // whole-frame replay register in the FPGA.
                            response_status <= STATUS_DUPLICATE_REPLAY;
                            control_state <= C_BUILD;
                        end else if (received_sequence != expected_sequence) begin
                            response_status <= STATUS_BAD_SEQUENCE;
                            if (sequence_errors != 16'hffff)
                                sequence_errors <= sequence_errors + 16'd1;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else if ((rx_frame[3] != CMD_EXECUTE) &&
                                     (rx_frame[3] != CMD_STATUS) &&
                                     (rx_frame[3] != CMD_CLEAR_COUNTERS)) begin
                            response_status <= STATUS_BAD_COMMAND;
                            error_strobe <= 1'b1;
                            control_state <= C_BUILD;
                        end else begin
                            for (byte_index = 0; byte_index < 24; byte_index = byte_index + 1)
                                request_payload[byte_index * 8 +: 8] <= rx_frame[12 + byte_index];
                            response_status <= STATUS_OK;
                            expected_sequence <= expected_sequence + 32'd1;
                            last_sequence <= received_sequence;
                            last_response_valid <= 1'b1;
                            control_state <= C_DISPATCH;
                        end
                    end else begin
                        rx_index <= rx_index + 6'd1;
                    end
                end
            end

            case (control_state)
                C_DISPATCH: begin
                    if (request_command == CMD_EXECUTE) begin
                        core_in_valid <= 1'b1;
                        response_start_cycle <= cycle_counter;
                        control_state <= C_EXEC_WAIT;
                    end else if (request_command == CMD_CLEAR_COUNTERS) begin
                        control_state <= C_CLEAR;
                    end else begin
                        capture_active_version <= active_version;
                        capture_debug_flags <= {
                            1'b0, host_commit_blocked, route_commit_pending,
                            route_selected_bank, active_bank, map_valid, commit_ack, 1'b0
                        };
                        control_state <= C_BUILD;
                    end
                end
                C_EXEC_WAIT: begin
                    // core_in_valid was asserted in C_DISPATCH and is sampled
                    // by the integrated core on this first C_EXEC_WAIT edge.
                    if (core_in_valid) begin
                        response_start_cycle <= cycle_counter;
                    end else if (route_action_word[0]) begin
                        response_action_cycle <= cycle_counter;
                        capture_out_word <= out_word;
                        capture_route_action_word <= route_action_word;
                        capture_route_state_word <= route_state_word;
                        capture_route_version_word <= route_version_word;
                        capture_map_llr <= map_llr;
                        capture_map_address <= map_address;
                        capture_active_version <= active_version;
                        capture_debug_flags <= {
                            1'b0, host_commit_blocked, route_commit_pending,
                            route_selected_bank, active_bank, map_valid, commit_ack, 1'b1
                        };
                        action_strobe <= 1'b1;
                        control_state <= C_BUILD;
                    end
                end
                C_CLEAR: begin
                    rx_crc_errors <= 16'd0;
                    sequence_errors <= 16'd0;
                    uart_framing_errors <= 16'd0;
                    frame_timeout_errors <= 16'd0;
                    dropped_rx_bytes <= 16'd0;
                    control_state <= C_BUILD;
                end
                C_BUILD: begin
                    response_frame <= 768'd0;
                    response_frame[7:0] <= 8'h5a;
                    response_frame[15:8] <= 8'ha5;
                    response_frame[23:16] <= PROTOCOL_VERSION;
                    response_frame[31:24] <= request_command;
                    response_frame[63:32] <= request_sequence;
                    response_frame[71:64] <= 8'd80;
                    response_frame[103:96] <= response_status;
                    response_frame[111:104] <= request_command;
                    response_frame[119:112] <= PROTOCOL_VERSION;
                    response_frame[127:120] <= capture_debug_flags;
                    response_frame[191:128] <= response_start_cycle;
                    response_frame[255:192] <= response_action_cycle;
                    response_frame[375:256] <= {2'd0, capture_out_word};
                    response_frame[455:376] <= capture_route_action_word;
                    response_frame[551:456] <= capture_route_state_word;
                    response_frame[615:552] <= capture_route_version_word;
                    response_frame[639:616] <= {2'd0, capture_map_llr};
                    response_frame[655:640] <= {7'd0, capture_map_address};
                    response_frame[671:656] <= capture_active_version;
                    response_frame[679:672] <= capture_debug_flags;
                    response_frame[687:680] <= 8'd0;
                    response_frame[703:688] <= rx_crc_errors;
                    response_frame[719:704] <= sequence_errors;
                    response_frame[735:720] <= uart_framing_errors;
                    response_crc <= 32'hffffffff;
                    response_crc_index <= 7'd0;
                    control_state <= C_CRC;
                end
                C_CRC: begin
                    response_crc <= response_crc_next;
                    if (response_crc_index == 7'd91) begin
                        response_frame[767:736] <= ~response_crc_next;
                        control_state <= C_READY;
                    end else begin
                        response_crc_index <= response_crc_index + 7'd1;
                    end
                end
                C_READY: begin
                    tx_index <= 7'd0;
                    control_state <= C_TX;
                end
                default: begin end
            endcase
        end
    end
endmodule
