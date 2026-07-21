// T6.25.2: production parameter-bank manager shared by the converged Route-A top.
//
// Image generation (bank*_image_version) is deliberately separated from the
// monotonic activation epoch carried by the core.  This permits fail-closed
// reactivation of an older trusted LKG image under a new activation epoch,
// without weakening compare-and-swap or allowing active-bank writes.

module gkp_parameter_bank_manager #(
    parameter integer WORDS_PER_PHASE = 257,
    parameter [3:0] RETIRED_BANK_DRAIN_CYCLES = 4'd6,
    parameter BANK0_INITIAL_TRUSTED = 1'b1,
    parameter BANK1_INITIAL_TRUSTED = 1'b1,
    parameter [15:0] BANK0_INITIAL_IMAGE_VERSION = 16'd0,
    parameter [15:0] BANK1_INITIAL_IMAGE_VERSION = 16'd1
) (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          safe_boundary,
    input  wire          core_active_bank,
    input  wire [15:0]   core_active_version,
    input  wire          core_commit_ack,

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

    input  wire          commit_request_valid,
    input  wire          commit_request_source_policy,
    input  wire          commit_request_bank,
    input  wire [15:0]   commit_expected_active_version,
    input  wire [15:0]   commit_new_activation_version,
    input  wire          commit_cancel_valid,
    input  wire          management_snapshot_request,

    output reg           cfg_begin_ack,
    output reg           cfg_word_ack,
    output reg           cfg_finalize_ack,
    output reg           cfg_abort_ack,
    output reg           commit_request_ack,
    output reg           commit_request_ack_source_policy,
    output reg           commit_complete,
    output reg           commit_complete_source_policy,
    output reg           commit_cancel_ack,
    output reg           management_snapshot_ack,
    output reg           management_state_valid,
    output reg           management_reject,
    output reg  [7:0]    management_reject_reason,
    output wire          management_ready,

    output wire          core_cfg_we,
    output wire          core_cfg_bank,
    output wire          core_cfg_phase,
    output wire [8:0]    core_cfg_address,
    output wire [21:0]   core_cfg_data,
    output wire          core_bank0_trusted,
    output wire          core_bank1_trusted,
    output wire          core_commit_valid,
    output wire          core_commit_bank,
    output wire [15:0]   core_commit_version,

    output wire          cfg_session_active_debug,
    output wire          cfg_staged_bank_debug,
    output wire [15:0]   cfg_staged_image_version_debug,
    output wire          cfg_next_phase_debug,
    output wire [8:0]    cfg_next_address_debug,
    output wire [9:0]    cfg_word_count_debug,
    output wire          cfg_all_words_received_debug,
    output wire          commit_pending_debug,
    output wire          commit_pending_bank_debug,
    output wire [15:0]   commit_pending_version_debug,
    output wire          commit_pending_source_policy_debug,
    output wire [3:0]    retired_bank_drain_count_debug,
    output wire [15:0]   bank0_image_version_debug,
    output wire [15:0]   bank1_image_version_debug,
    output wire [31:0]   cfg_running_crc32_debug,
    output wire [31:0]   cfg_expected_crc32_debug,
    output wire [159:0]  management_state_word
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

    function automatic [31:0] crc32_byte(input [31:0] crc_in, input [7:0] octet);
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

    function automatic [31:0] crc32_word22(input [31:0] crc_in, input [21:0] word_value);
        reg [31:0] crc;
        begin
            crc = crc32_byte(crc_in, word_value[7:0]);
            crc = crc32_byte(crc, word_value[15:8]);
            crc32_word22 = crc32_byte(crc, {2'b00, word_value[21:16]});
        end
    endfunction

    function automatic [15:0] crc16_byte(input [15:0] crc_in, input [7:0] octet);
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
    reg [15:0] bank0_image_version;
    reg [15:0] bank1_image_version;

    reg cfg_session_active;
    reg cfg_staged_bank;
    reg [15:0] cfg_staged_image_version;
    reg [31:0] cfg_staged_expected_crc32;
    reg [31:0] cfg_running_crc32;
    reg cfg_next_phase;
    reg [8:0] cfg_next_address;
    reg [9:0] cfg_word_count;
    reg cfg_all_words_received;

    reg commit_pending;
    reg commit_pending_bank;
    reg [15:0] commit_pending_version;
    reg commit_pending_source_policy;
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
    wire management_conflict = management_request_count > 3'd1;
    wire selected_commit_trusted = commit_request_bank ? bank1_trusted : bank0_trusted;
    wire [15:0] maximum_image_version =
        (bank0_image_version > bank1_image_version) ? bank0_image_version : bank1_image_version;
    wire cfg_word_exact = cfg_session_active &&
                          (cfg_word_phase == cfg_next_phase) &&
                          (cfg_word_address == cfg_next_address) &&
                          (cfg_word_address < WORDS_PER_PHASE) &&
                          (cfg_staged_bank != core_active_bank) &&
                          !commit_pending;

    assign core_cfg_we = cfg_word_valid && !management_conflict && cfg_word_exact;
    assign core_cfg_bank = cfg_staged_bank;
    assign core_cfg_phase = cfg_word_phase;
    assign core_cfg_address = cfg_word_address;
    assign core_cfg_data = cfg_word_data;
    assign core_bank0_trusted = bank0_trusted;
    assign core_bank1_trusted = bank1_trusted;
    // The core acknowledges on the clock after accepting a commit.  Re-check
    // the complete acceptance guard here so the pending request is not
    // re-presented after the active bank/version have already advanced but
    // before the registered acknowledgement is observed by this manager.
    assign core_commit_valid = commit_pending && safe_boundary &&
                               (commit_pending_bank != core_active_bank) &&
                               (core_active_version != 16'hffff) &&
                               (commit_pending_version == core_active_version + 16'd1) &&
                               (commit_pending_bank ? bank1_trusted : bank0_trusted);
    assign core_commit_bank = commit_pending_bank;
    assign core_commit_version = commit_pending_version;
    assign management_ready = !cfg_session_active && !commit_pending &&
                              !management_snapshot_busy &&
                              (retired_bank_drain_count == 4'd0);

    assign cfg_session_active_debug = cfg_session_active;
    assign cfg_staged_bank_debug = cfg_staged_bank;
    assign cfg_staged_image_version_debug = cfg_staged_image_version;
    assign cfg_next_phase_debug = cfg_next_phase;
    assign cfg_next_address_debug = cfg_next_address;
    assign cfg_word_count_debug = cfg_word_count;
    assign cfg_all_words_received_debug = cfg_all_words_received;
    assign commit_pending_debug = commit_pending;
    assign commit_pending_bank_debug = commit_pending_bank;
    assign commit_pending_version_debug = commit_pending_version;
    assign commit_pending_source_policy_debug = commit_pending_source_policy;
    assign retired_bank_drain_count_debug = retired_bank_drain_count;
    assign bank0_image_version_debug = bank0_image_version;
    assign bank1_image_version_debug = bank1_image_version;
    assign cfg_running_crc32_debug = cfg_running_crc32;
    assign cfg_expected_crc32_debug = cfg_staged_expected_crc32;

    reg [143:0] management_state_payload;
    always @* begin
        management_state_payload = 144'd0;
        management_state_payload[0] = core_active_bank;
        management_state_payload[16:1] = core_active_version;
        management_state_payload[17] = bank0_trusted;
        management_state_payload[18] = bank1_trusted;
        management_state_payload[34:19] = bank0_image_version;
        management_state_payload[50:35] = bank1_image_version;
        management_state_payload[51] = cfg_session_active;
        management_state_payload[52] = cfg_staged_bank;
        management_state_payload[62:53] = cfg_word_count;
        management_state_payload[63] = cfg_next_phase;
        management_state_payload[72:64] = cfg_next_address;
        management_state_payload[73] = cfg_all_words_received;
        management_state_payload[74] = commit_pending;
        management_state_payload[75] = commit_pending_bank;
        management_state_payload[91:76] = commit_pending_version;
        management_state_payload[92] = commit_pending_source_policy;
        management_state_payload[96:93] = retired_bank_drain_count;
        management_state_payload[104:97] = management_reject_reason;
        management_state_payload[120:105] = management_reject_count;
        management_state_payload[136:121] = crc_failure_count;
        management_state_payload[137] = protocol_fault_sticky;
        management_state_payload[138] = management_snapshot_busy;
    end
    wire [7:0] management_snapshot_octet = management_snapshot_shift[7:0];
    wire [15:0] management_snapshot_crc_next =
        crc16_byte(management_snapshot_crc, management_snapshot_octet);
    assign management_state_word = management_state_word_reg;

    always @(posedge clk) begin
        if (!reset_n) begin
            bank0_trusted <= BANK0_INITIAL_TRUSTED;
            bank1_trusted <= BANK1_INITIAL_TRUSTED;
            bank0_image_version <= BANK0_INITIAL_IMAGE_VERSION;
            bank1_image_version <= BANK1_INITIAL_IMAGE_VERSION;
            cfg_session_active <= 1'b0;
            cfg_staged_bank <= 1'b0;
            cfg_staged_image_version <= 16'd0;
            cfg_staged_expected_crc32 <= 32'd0;
            cfg_running_crc32 <= 32'hffffffff;
            cfg_next_phase <= 1'b0;
            cfg_next_address <= 9'd0;
            cfg_word_count <= 10'd0;
            cfg_all_words_received <= 1'b0;
            commit_pending <= 1'b0;
            commit_pending_bank <= 1'b0;
            commit_pending_version <= 16'd0;
            commit_pending_source_policy <= 1'b0;
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
            commit_request_ack_source_policy <= 1'b0;
            commit_complete <= 1'b0;
            commit_complete_source_policy <= 1'b0;
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
            commit_request_ack_source_policy <= 1'b0;
            commit_complete <= 1'b0;
            commit_complete_source_policy <= 1'b0;
            commit_cancel_ack <= 1'b0;
            management_snapshot_ack <= 1'b0;
            management_state_valid <= 1'b0;
            management_reject <= 1'b0;
            management_reject_reason <= REJECT_NONE;

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
                    management_snapshot_byte_index <= management_snapshot_byte_index + 5'd1;
                end
            end

            if (retired_bank_drain_count != 4'd0)
                retired_bank_drain_count <= retired_bank_drain_count - 4'd1;

            if (core_commit_ack) begin
                commit_pending <= 1'b0;
                commit_complete <= 1'b1;
                commit_complete_source_policy <= commit_pending_source_policy;
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
                if (cfg_session_active || commit_pending || management_snapshot_busy) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_BUSY;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (retired_bank_drain_count != 4'd0) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_DRAIN_GUARD;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (cfg_begin_bank == core_active_bank) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_ACTIVE_BANK;
                    management_reject_count <= sat16_inc(management_reject_count);
                    protocol_fault_sticky <= 1'b1;
                end else if ((core_active_version == 16'hffff) ||
                             (cfg_expected_active_version != core_active_version) ||
                             (maximum_image_version == 16'hffff) ||
                             (cfg_new_image_version <= maximum_image_version)) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_VERSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else begin
                    cfg_session_active <= 1'b1;
                    cfg_staged_bank <= cfg_begin_bank;
                    cfg_staged_image_version <= cfg_new_image_version;
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
                    if (!cfg_next_phase && (cfg_next_address == WORDS_PER_PHASE - 1)) begin
                        cfg_next_phase <= 1'b1;
                        cfg_next_address <= 9'd0;
                    end else if (cfg_next_phase && (cfg_next_address == WORDS_PER_PHASE - 1)) begin
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
                end else if (!cfg_all_words_received ||
                             (cfg_word_count != (2 * WORDS_PER_PHASE))) begin
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
                end else if ((cfg_running_crc32 ^ 32'hffffffff) != cfg_staged_expected_crc32) begin
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
                        bank1_image_version <= cfg_staged_image_version;
                    end else begin
                        bank0_trusted <= 1'b1;
                        bank0_image_version <= cfg_staged_image_version;
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
                if (cfg_session_active || commit_pending || management_snapshot_busy) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_BUSY;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (retired_bank_drain_count != 4'd0) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_DRAIN_GUARD;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (commit_request_bank == core_active_bank) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_ACTIVE_BANK;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if (!selected_commit_trusted) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_UNTRUSTED;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else if ((core_active_version == 16'hffff) ||
                             (commit_expected_active_version != core_active_version) ||
                             (commit_new_activation_version != (core_active_version + 16'd1))) begin
                    management_reject <= 1'b1;
                    management_reject_reason <= REJECT_VERSION;
                    management_reject_count <= sat16_inc(management_reject_count);
                end else begin
                    commit_pending <= 1'b1;
                    commit_pending_bank <= commit_request_bank;
                    commit_pending_version <= commit_new_activation_version;
                    commit_pending_source_policy <= commit_request_source_policy;
                    commit_request_ack <= 1'b1;
                    commit_request_ack_source_policy <= commit_request_source_policy;
                end
            end
        end
    end
endmodule
