// T6.2.1: board-independent production management shell for the six-cycle
// GKP fast path.  The transport/CDC layer is intentionally outside this file.
//
// Contract highlights:
//   * only the inactive A/B bank can be programmed;
//   * exactly X[0:256], then Z[0:256], is accepted;
//   * CRC32/IEEE covers each 22-bit word as three little-endian bytes;
//   * commit uses compare-and-swap versions and waits for safe_boundary;
//   * a six-cycle drain guard protects requests already using the retired bank;
//   * any protocol/CRC fault leaves the staged bank untrusted.

module gkp_fast_path_production_top #(
    parameter [7:0]  OOD_THRESHOLD_CODE       = 8'd192,
    parameter [15:0] MAX_PARAMETER_AGE_CYCLES = 16'd8192,
    parameter [3:0]  RETIRED_BANK_DRAIN_CYCLES = 4'd6,
    parameter        BANK0_INITIAL_TRUSTED = 1'b1,
    parameter        BANK1_INITIAL_TRUSTED = 1'b1,
    parameter [15:0] BANK0_INITIAL_VERSION = 16'd0,
    parameter [15:0] BANK1_INITIAL_VERSION = 16'd1
) (
    input  wire          clk,
    input  wire          reset_n,

    input  wire          in_valid,
    input  wire [57:0]   in_word,
    input  wire          safe_boundary,

    input  wire          cfg_begin_valid,
    input  wire          cfg_begin_bank,
    input  wire [15:0]   cfg_expected_active_version,
    input  wire [15:0]   cfg_new_version,
    input  wire [31:0]   cfg_expected_crc32,
    input  wire          cfg_word_valid,
    input  wire          cfg_word_phase,
    input  wire [8:0]    cfg_word_address,
    input  wire [21:0]   cfg_word_data,
    input  wire          cfg_finalize_valid,
    input  wire          cfg_abort_valid,

    input  wire          commit_request_valid,
    input  wire          commit_request_bank,
    input  wire [15:0]   commit_expected_active_version,
    input  wire [15:0]   commit_new_version,
    input  wire          commit_cancel_valid,
    input  wire          management_snapshot_request,

    output reg           cfg_begin_ack,
    output reg           cfg_word_ack,
    output reg           cfg_finalize_ack,
    output reg           cfg_abort_ack,
    output reg           commit_request_ack,
    output reg           commit_complete,
    output reg           commit_cancel_ack,
    output reg           management_snapshot_ack,
    output reg           management_state_valid,
    output reg           management_reject,
    output reg  [7:0]    management_reject_reason,
    output wire          cfg_session_active_debug,
    output wire          commit_pending_debug,
    output wire          management_snapshot_busy_debug,
    output wire [159:0]  management_state_word,

    output wire [117:0]  out_word,
    output wire [231:0]  state_word,
    output wire          map_valid_debug,
    output wire [21:0]   map_llr_debug,
    output wire [8:0]    map_address_debug,
    output wire [15:0]   active_version_debug,
    output wire          active_bank_debug
);

    localparam [7:0] REJECT_NONE          = 8'h00;
    localparam [7:0] REJECT_CONFLICT      = 8'h01;
    localparam [7:0] REJECT_BUSY          = 8'h02;
    localparam [7:0] REJECT_ACTIVE_BANK   = 8'h03;
    localparam [7:0] REJECT_VERSION       = 8'h04;
    localparam [7:0] REJECT_DRAIN_GUARD   = 8'h05;
    localparam [7:0] REJECT_NO_SESSION    = 8'h06;
    localparam [7:0] REJECT_WORD_ORDER    = 8'h07;
    localparam [7:0] REJECT_CRC32         = 8'h08;
    localparam [7:0] REJECT_INCOMPLETE    = 8'h09;
    localparam [7:0] REJECT_NO_PENDING    = 8'h0a;
    localparam [7:0] REJECT_UNTRUSTED     = 8'h0b;

    function automatic [15:0] sat16_inc(input [15:0] value);
        begin
            sat16_inc = (value == 16'hffff) ? value : value + 16'd1;
        end
    endfunction

    // Reflected CRC32/IEEE (zlib-compatible): init 0xffffffff, final xor.
    function automatic [31:0] crc32_byte(
        input [31:0] crc_in,
        input [7:0] octet
    );
        integer bit_index;
        reg [31:0] crc;
        reg [7:0] data;
        begin
            crc = crc_in;
            data = octet;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                if (crc[0] ^ data[0])
                    crc = (crc >> 1) ^ 32'hedb88320;
                else
                    crc = crc >> 1;
                data = data >> 1;
            end
            crc32_byte = crc;
        end
    endfunction

    function automatic [31:0] crc32_word22(
        input [31:0] crc_in,
        input [21:0] word_value
    );
        reg [31:0] crc;
        begin
            crc = crc32_byte(crc_in, word_value[7:0]);
            crc = crc32_byte(crc, word_value[15:8]);
            crc = crc32_byte(crc, {2'b00, word_value[21:16]});
            crc32_word22 = crc;
        end
    endfunction

    function automatic [15:0] crc16_byte(
        input [15:0] crc_in,
        input [7:0] octet
    );
        integer bit_index;
        reg [15:0] crc;
        begin
            crc = crc_in ^ ({8'd0, octet} << 8);
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            crc16_byte = crc;
        end
    endfunction

    reg bank0_trusted;
    reg bank1_trusted;
    reg [15:0] bank0_version;
    reg [15:0] bank1_version;

    reg cfg_session_active;
    reg cfg_staged_bank;
    reg [15:0] cfg_staged_version;
    reg [31:0] cfg_staged_expected_crc32;
    reg [31:0] cfg_running_crc32;
    reg cfg_next_phase;
    reg [8:0] cfg_next_address;
    reg [9:0] cfg_word_count;
    reg cfg_all_words_received;

    reg commit_pending;
    reg commit_pending_bank;
    reg [15:0] commit_pending_version;
    reg [3:0] retired_bank_drain_count;

    reg protocol_fault_sticky;
    reg [15:0] management_reject_count;
    reg [15:0] crc_failure_count;
    reg management_snapshot_busy;
    reg [4:0] management_snapshot_byte_index;
    reg [143:0] management_snapshot_payload;
    reg [143:0] management_snapshot_shift;
    reg [15:0] management_snapshot_crc;
    reg [159:0] management_state_word_reg;

    wire [2:0] management_request_count =
        {2'b00, cfg_begin_valid} + {2'b00, cfg_word_valid} +
        {2'b00, cfg_finalize_valid} + {2'b00, cfg_abort_valid} +
        {2'b00, commit_request_valid} + {2'b00, commit_cancel_valid} +
        {2'b00, management_snapshot_request};
    wire management_conflict = (management_request_count > 3'd1);

    wire selected_staged_trusted = cfg_staged_bank ? bank1_trusted : bank0_trusted;
    wire selected_commit_trusted = commit_request_bank ? bank1_trusted : bank0_trusted;
    wire [15:0] selected_commit_version = commit_request_bank ?
                                                bank1_version : bank0_version;
    wire cfg_word_exact = cfg_session_active &&
                          (cfg_word_phase == cfg_next_phase) &&
                          (cfg_word_address == cfg_next_address) &&
                          (cfg_word_address <= 9'd256) &&
                          (cfg_staged_bank != active_bank_debug) &&
                          !commit_pending;

    wire core_cfg_we = cfg_word_valid && !management_conflict && cfg_word_exact;
    wire core_commit_valid = commit_pending && safe_boundary;
    wire core_commit_ack;

    assign cfg_session_active_debug = cfg_session_active;
    assign commit_pending_debug = commit_pending;
    assign management_snapshot_busy_debug = management_snapshot_busy;

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
        .commit_bank(commit_pending_bank),
        .commit_version(commit_pending_version),
        .commit_ack(core_commit_ack),
        .cfg_we(core_cfg_we),
        .cfg_bank(cfg_staged_bank),
        .cfg_phase(cfg_word_phase),
        .cfg_address(cfg_word_address),
        .cfg_data(cfg_word_data),
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

    reg [143:0] management_state_payload;
    always @* begin
        management_state_payload = 144'd0;
        management_state_payload[0] = active_bank_debug;
        management_state_payload[16:1] = active_version_debug;
        management_state_payload[17] = bank0_trusted;
        management_state_payload[18] = bank1_trusted;
        management_state_payload[34:19] = bank0_version;
        management_state_payload[50:35] = bank1_version;
        management_state_payload[51] = cfg_session_active;
        management_state_payload[52] = cfg_staged_bank;
        management_state_payload[62:53] = cfg_word_count;
        management_state_payload[63] = cfg_next_phase;
        management_state_payload[72:64] = cfg_next_address;
        management_state_payload[73] = cfg_all_words_received;
        management_state_payload[74] = commit_pending;
        management_state_payload[75] = commit_pending_bank;
        management_state_payload[91:76] = commit_pending_version;
        management_state_payload[95:92] = retired_bank_drain_count;
        management_state_payload[103:96] = management_reject_reason;
        management_state_payload[119:104] = management_reject_count;
        management_state_payload[135:120] = crc_failure_count;
        management_state_payload[136] = protocol_fault_sticky;
        management_state_payload[137] = management_snapshot_busy;
    end
    wire [7:0] management_snapshot_octet = management_snapshot_shift[7:0];
    wire [15:0] management_snapshot_crc_next =
        crc16_byte(management_snapshot_crc, management_snapshot_octet);
    assign management_state_word = management_state_word_reg;

    always @(posedge clk) begin
        if (!reset_n) begin
            bank0_trusted <= BANK0_INITIAL_TRUSTED;
            bank1_trusted <= BANK1_INITIAL_TRUSTED;
            bank0_version <= BANK0_INITIAL_VERSION;
            bank1_version <= BANK1_INITIAL_VERSION;
            cfg_session_active <= 1'b0;
            cfg_staged_bank <= 1'b0;
            cfg_staged_version <= 16'd0;
            cfg_staged_expected_crc32 <= 32'd0;
            cfg_running_crc32 <= 32'hffffffff;
            cfg_next_phase <= 1'b0;
            cfg_next_address <= 9'd0;
            cfg_word_count <= 10'd0;
            cfg_all_words_received <= 1'b0;
            commit_pending <= 1'b0;
            commit_pending_bank <= 1'b0;
            commit_pending_version <= 16'd0;
            retired_bank_drain_count <= 4'd0;
            protocol_fault_sticky <= 1'b0;
            management_reject_count <= 16'd0;
            crc_failure_count <= 16'd0;
            management_snapshot_busy <= 1'b0;
            management_snapshot_byte_index <= 5'd0;
            management_snapshot_payload <= 144'd0;
            management_snapshot_shift <= 144'd0;
            management_snapshot_crc <= 16'hffff;
            management_state_word_reg <= 160'd0;
            cfg_begin_ack <= 1'b0;
            cfg_word_ack <= 1'b0;
            cfg_finalize_ack <= 1'b0;
            cfg_abort_ack <= 1'b0;
            commit_request_ack <= 1'b0;
            commit_complete <= 1'b0;
            commit_cancel_ack <= 1'b0;
            management_snapshot_ack <= 1'b0;
            management_state_valid <= 1'b0;
            management_reject <= 1'b0;
            management_reject_reason <= REJECT_NONE;
        end else begin
            cfg_begin_ack <= 1'b0;
            cfg_word_ack <= 1'b0;
            cfg_finalize_ack <= 1'b0;
            cfg_abort_ack <= 1'b0;
            commit_request_ack <= 1'b0;
            commit_complete <= 1'b0;
            commit_cancel_ack <= 1'b0;
            management_snapshot_ack <= 1'b0;
            management_state_valid <= 1'b0;
            management_reject <= 1'b0;

            // Snapshot CRC is deliberately byte-serial.  A 144-bit
            // combinational CRC passed semantics but created a long, costly
            // management-only path; 18 bounded cycles keep it off fast path.
            if (management_snapshot_busy) begin
                management_snapshot_crc <= management_snapshot_crc_next;
                management_snapshot_shift <= management_snapshot_shift >> 8;
                if (management_snapshot_byte_index == 5'd17) begin
                    management_state_word_reg <= {
                        management_snapshot_crc_next,
                        management_snapshot_payload
                    };
                    management_snapshot_busy <= 1'b0;
                    management_state_valid <= 1'b1;
                end else begin
                    management_snapshot_byte_index <=
                        management_snapshot_byte_index + 5'd1;
                end
            end

            if (retired_bank_drain_count != 4'd0)
                retired_bank_drain_count <= retired_bank_drain_count - 4'd1;

            if (core_commit_ack) begin
                commit_pending <= 1'b0;
                commit_complete <= 1'b1;
                retired_bank_drain_count <= RETIRED_BANK_DRAIN_CYCLES;
            end

            if (management_conflict) begin
                management_reject <= 1'b1;
                management_reject_reason <= REJECT_CONFLICT;
                management_reject_count <= sat16_inc(management_reject_count);
                protocol_fault_sticky <= 1'b1;
                if (cfg_session_active) begin
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    if (cfg_staged_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end
                commit_pending <= 1'b0;
            end else if (management_snapshot_request) begin
                if (management_snapshot_busy) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_BUSY;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else begin
                    management_snapshot_payload <= management_state_payload;
                    management_snapshot_shift <= management_state_payload;
                    management_snapshot_crc <= 16'hffff;
                    management_snapshot_byte_index <= 5'd0;
                    management_snapshot_busy <= 1'b1;
                    management_snapshot_ack <= 1'b1;
                end
            end else if (cfg_abort_valid) begin
                if (cfg_session_active) begin
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    cfg_abort_ack <= 1'b1;
                    if (cfg_staged_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end else begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_NO_SESSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                    protocol_fault_sticky <= 1'b1;
                end
            end else if (cfg_begin_valid) begin
                if (cfg_session_active || commit_pending) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_BUSY;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (retired_bank_drain_count != 4'd0) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_DRAIN_GUARD;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (cfg_begin_bank == active_bank_debug) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_ACTIVE_BANK;
                    management_reject_count <= sat16_inc(management_reject_count);
                    protocol_fault_sticky <= 1'b1;
                end else if ((active_version_debug == 16'hffff) ||
                             (cfg_expected_active_version != active_version_debug) ||
                             (cfg_new_version != (active_version_debug + 16'd1))) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_VERSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else begin
                    cfg_session_active <= 1'b1;
                    cfg_staged_bank <= cfg_begin_bank;
                    cfg_staged_version <= cfg_new_version;
                    cfg_staged_expected_crc32 <= cfg_expected_crc32;
                    cfg_running_crc32 <= 32'hffffffff;
                    cfg_next_phase <= 1'b0;
                    cfg_next_address <= 9'd0;
                    cfg_word_count <= 10'd0;
                    cfg_all_words_received <= 1'b0;
                    cfg_begin_ack <= 1'b1;
                    if (cfg_begin_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end
            end else if (cfg_word_valid) begin
                if (!cfg_session_active) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_NO_SESSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                    protocol_fault_sticky <= 1'b1;
                end else if (!cfg_word_exact || cfg_all_words_received) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_WORD_ORDER;
                    management_reject_count <= sat16_inc(management_reject_count);
                    protocol_fault_sticky <= 1'b1;
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    if (cfg_staged_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end else begin
                    cfg_running_crc32 <= crc32_word22(cfg_running_crc32, cfg_word_data);
                    cfg_word_count <= cfg_word_count + 10'd1;
                    cfg_word_ack <= 1'b1;
                    if (!cfg_next_phase && (cfg_next_address == 9'd256)) begin
                        cfg_next_phase <= 1'b1;
                        cfg_next_address <= 9'd0;
                    end else if (cfg_next_phase && (cfg_next_address == 9'd256)) begin
                        cfg_all_words_received <= 1'b1;
                    end else begin
                        cfg_next_address <= cfg_next_address + 9'd1;
                    end
                end
            end else if (cfg_finalize_valid) begin
                if (!cfg_session_active) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_NO_SESSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (!cfg_all_words_received || (cfg_word_count != 10'd514)) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_INCOMPLETE;
                    management_reject_count <= sat16_inc(management_reject_count);
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    protocol_fault_sticky <= 1'b1;
                    if (cfg_staged_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end else if ((cfg_running_crc32 ^ 32'hffffffff) !=
                             cfg_staged_expected_crc32) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_CRC32;
                    management_reject_count <= sat16_inc(management_reject_count);
                    crc_failure_count <= sat16_inc(crc_failure_count);
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    protocol_fault_sticky <= 1'b1;
                    if (cfg_staged_bank)
                        bank1_trusted <= 1'b0;
                    else
                        bank0_trusted <= 1'b0;
                end else begin
                    cfg_session_active <= 1'b0;
                    cfg_all_words_received <= 1'b0;
                    cfg_finalize_ack <= 1'b1;
                    if (cfg_staged_bank) begin
                        bank1_trusted <= 1'b1;
                        bank1_version <= cfg_staged_version;
                    end else begin
                        bank0_trusted <= 1'b1;
                        bank0_version <= cfg_staged_version;
                    end
                end
            end else if (commit_cancel_valid) begin
                if (commit_pending) begin
                    commit_pending <= 1'b0;
                    commit_cancel_ack <= 1'b1;
                end else begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_NO_PENDING;
                    management_reject_count <= sat16_inc(management_reject_count);
                end
            end else if (commit_request_valid) begin
                if (cfg_session_active || commit_pending) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_BUSY;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (retired_bank_drain_count != 4'd0) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_DRAIN_GUARD;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (commit_request_bank == active_bank_debug) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_ACTIVE_BANK;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (!selected_commit_trusted) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_UNTRUSTED;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if ((active_version_debug == 16'hffff) ||
                             (commit_expected_active_version != active_version_debug) ||
                             (commit_new_version != (active_version_debug + 16'd1)) ||
                             (selected_commit_version != commit_new_version)) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_VERSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else begin
                    commit_pending <= 1'b1;
                    commit_pending_bank <= commit_request_bank;
                    commit_pending_version <= commit_new_version;
                    commit_request_ack <= 1'b1;
                end
            end
        end
    end

endmodule
