// Small-pin activity harness for target-device P&R only.  It keeps every core
// output observable through a rolling signature without pretending to be the
// T6.1 serial transport.  Core cycle latency excludes this harness.

module gkp_fast_path_synth_top (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] seed_in,
    output wire [7:0] status_out
);
    reg [31:0] lfsr;
    reg [12:0] cycle_counter;
    reg [31:0] signature;
    reg [31:0] fold0, fold1, fold2, fold3, fold4, fold5, fold6;
    reg [41:0] payload;
    wire [57:0] input_word;
    wire [117:0] output_word;
    wire [231:0] state_word;
    wire [21:0] map_llr;
    wire [8:0] map_address;
    wire [15:0] active_version;
    wire map_valid;
    wire commit_ack;
    wire cfg_activity = (cycle_counter[11:0] == 12'hfff);
    wire cfg_inactive_bank = ~active_version[0];
    // The logical tables contain addresses 0..256. Keep the activity-only
    // write path inside that range for every seed instead of exposing the
    // 257-entry memories to raw 9-bit LFSR addresses 257..511.
    wire [8:0] cfg_safe_address =
        (lfsr[8:0] <= 9'd256) ? lfsr[8:0] : (lfsr[8:0] - 9'd257);

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
        payload[39] = 1'b0;
        payload[40] = 1'b1;
        payload[41] = 1'b1;
    end

    assign input_word = {crc16_42(payload), payload};
    assign status_out = {signature[5:0], commit_ack, map_valid};

    always @(posedge clk) begin
        if (!reset_n) begin
            lfsr <= {24'h1aceb0, seed_in ^ 8'h5a};
            cycle_counter <= 13'd0;
            signature <= 32'h6d2b79f5;
            fold0 <= 32'd0;
            fold1 <= 32'd0;
            fold2 <= 32'd0;
            fold3 <= 32'd0;
            fold4 <= 32'd0;
            fold5 <= 32'd0;
            fold6 <= 32'd0;
        end else begin
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            cycle_counter <= cycle_counter + 13'd1;
            // Segment registers bound harness depth.  A round-robin consumer
            // keeps every core output observable without putting a 350-bit
            // XOR tree on the core timing path.
            fold0 <= output_word[31:0] ^ output_word[63:32];
            fold1 <= output_word[95:64] ^ {10'd0, output_word[117:96]};
            fold2 <= state_word[31:0] ^ state_word[63:32];
            fold3 <= state_word[95:64] ^ state_word[127:96];
            fold4 <= state_word[159:128] ^ state_word[191:160];
            fold5 <= state_word[223:192] ^ {24'd0, state_word[231:224]};
            fold6 <= {1'b0, active_version, map_address, map_llr[5:0]};
            case (cycle_counter[2:0])
                3'd0: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold0;
                3'd1: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold1;
                3'd2: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold2;
                3'd3: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold3;
                3'd4: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold4;
                3'd5: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold5;
                default: signature <= {signature[30:0], signature[31] ^ signature[21]} ^ fold6;
            endcase
        end
    end

    gkp_fast_path_core core (
        .clk(clk),
        .reset_n(reset_n),
        .in_valid(1'b1),
        .in_word(input_word),
        .safe_boundary(1'b1),
        .commit_valid(cycle_counter == 13'd4000),
        .commit_bank(1'b1),
        .commit_version(16'd1),
        .commit_ack(commit_ack),
        // A sparse write to the inactive bank keeps the real configuration
        // write ports and all signed-22 lanes in the synthesis image.  It is
        // activity-harness traffic, not a deployment update protocol.
        .cfg_we(cfg_activity),
        .cfg_bank(cfg_inactive_bank),
        .cfg_phase(lfsr[17]),
        .cfg_address(cfg_safe_address),
        .cfg_data(lfsr[21:0]),
        .bank0_trusted(1'b1),
        .bank1_trusted(1'b1),
        .out_word(output_word),
        .state_word(state_word),
        .map_valid_debug(map_valid),
        .map_llr_debug(map_llr),
        .map_address_debug(map_address),
        .active_version_debug(active_version)
    );
endmodule
