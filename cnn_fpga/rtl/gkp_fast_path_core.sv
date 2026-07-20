// T-RISK-20260716-01: synthesizable counterpart of the T5.5.1 Python golden.
// The host verifies the binary image CRC32/SHA256 before asserting bank_trusted.
// This core implements the per-cycle data path, safe-boundary commit, MAP-LUT,
// fallback/event FSM, frame state, and CRC-protected output/state words.

module gkp_fast_path_core #(
    // T5.5 legacy defaults are retained for bit-exact regression.  The
    // production management top overrides the age/version ceilings.
    parameter [7:0]  OOD_THRESHOLD_CODE       = 8'd192,
    parameter [15:0] MAX_PARAMETER_AGE_CYCLES = 16'd64,
    parameter [15:0] MAX_TRUSTED_BANK_VERSION = 16'd7
) (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          in_valid,
    input  wire [57:0]   in_word,
    input  wire          safe_boundary,

    input  wire          commit_valid,
    input  wire          commit_bank,
    input  wire [15:0]   commit_version,
    output reg           commit_ack,

    input  wire          cfg_we,
    input  wire          cfg_bank,
    input  wire          cfg_phase,
    input  wire [8:0]    cfg_address,
    input  wire [21:0]   cfg_data,
    input  wire          bank0_trusted,
    input  wire          bank1_trusted,

    output wire [117:0]  out_word,
    output wire [231:0]  state_word,
    output reg           map_valid_debug,
    output reg  [21:0]   map_llr_debug,
    output reg  [8:0]    map_address_debug,
    output wire [15:0]   active_version_debug,
    output wire          active_bank_debug
);

    localparam signed [25:0] LLR_MIN = -26'sd2097152;
    localparam signed [25:0] LLR_MAX =  26'sd2097151;

    localparam [2:0] MODE_NORMAL        = 3'd0;
    localparam [2:0] MODE_X_RECOVERY    = 3'd1;
    localparam [2:0] MODE_Z_RECOVERY    = 3'd2;
    localparam [2:0] MODE_HOLD          = 3'd3;
    localparam [2:0] MODE_RESET_REQUEST = 3'd4;
    localparam [2:0] MODE_FALLBACK      = 3'd5;

    localparam [2:0] HEALTH_HEALTHY        = 3'd0;
    localparam [2:0] HEALTH_DEGRADED       = 3'd1;
    localparam [2:0] HEALTH_RECOVERING      = 3'd2;
    localparam [2:0] HEALTH_FALLBACK        = 3'd3;
    localparam [2:0] HEALTH_RESET_REQUIRED  = 3'd4;

    function automatic [2:0] sat3_inc(input [2:0] value);
        begin
            sat3_inc = (value == 3'd7) ? value : value + 3'd1;
        end
    endfunction

    function automatic [7:0] sat8_inc(input [7:0] value);
        begin
            sat8_inc = (value == 8'hff) ? value : value + 8'd1;
        end
    endfunction

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

    function automatic [15:0] crc16_102(input [101:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [103:0] padded;
        reg [7:0] octet;
        begin
            padded = {2'b0, payload};
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 13; byte_index = byte_index + 1) begin
                octet = padded[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_102 = crc;
        end
    endfunction

    function automatic [15:0] crc16_216(input [215:0] payload);
        integer byte_index;
        integer bit_index;
        reg [15:0] crc;
        reg [7:0] octet;
        begin
            crc = 16'hffff;
            for (byte_index = 0; byte_index < 27; byte_index = byte_index + 1) begin
                octet = payload[byte_index * 8 +: 8];
                crc = crc ^ {octet, 8'h00};
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                    crc = crc[15] ? ((crc << 1) ^ 16'h1021) : (crc << 1);
            end
            crc16_216 = crc;
        end
    endfunction

    function automatic signed [24:0] round_shift3_ties_even(
        input signed [27:0] value
    );
        reg negative;
        reg [27:0] magnitude;
        reg [24:0] quotient;
        reg [2:0] remainder;
        begin
            negative = value[27];
            magnitude = negative ? -value : value;
            quotient = magnitude >> 3;
            remainder = magnitude[2:0];
            if ((remainder > 3'd4) || ((remainder == 3'd4) && quotient[0]))
                quotient = quotient + 25'd1;
            round_shift3_ties_even = negative ? -$signed(quotient) : $signed(quotient);
        end
    endfunction

    function automatic [21:0] saturate_llr(input signed [25:0] value);
        begin
            if (value > LLR_MAX)
                saturate_llr = 22'h1fffff;
            else if (value < LLR_MIN)
                saturate_llr = 22'h200000;
            else
                saturate_llr = value[21:0];
        end
    endfunction

    // Each logical table is mirrored.  One physical copy serves y0 and one
    // serves y1, so every inferred BSRAM has one synchronous read plus the
    // broadcast configuration write.  A non-mirrored 2R+1W memory cannot map
    // to the GW2AR BSRAM primitive while preserving II=1.
    (* ram_style = "block" *) reg [21:0] bank0_x_y0 [0:256];
    (* ram_style = "block" *) reg [21:0] bank0_x_y1 [0:256];
    (* ram_style = "block" *) reg [21:0] bank0_z_y0 [0:256];
    (* ram_style = "block" *) reg [21:0] bank0_z_y1 [0:256];
    (* ram_style = "block" *) reg [21:0] bank1_x_y0 [0:256];
    (* ram_style = "block" *) reg [21:0] bank1_x_y1 [0:256];
    (* ram_style = "block" *) reg [21:0] bank1_z_y0 [0:256];
    (* ram_style = "block" *) reg [21:0] bank1_z_y1 [0:256];

    initial begin
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank0_x.mem", bank0_x_y0);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank0_x.mem", bank0_x_y1);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank0_z.mem", bank0_z_y0);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank0_z.mem", bank0_z_y1);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank1_x.mem", bank1_x_y0);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank1_x.mem", bank1_x_y1);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank1_z.mem", bank1_z_y0);
        $readmemh("cnn_fpga/rtl/generated/t5_5_1_bank1_z.mem", bank1_z_y1);
    end

    reg active_bank;
    reg [15:0] active_version;
    assign active_version_debug = active_version;
    assign active_bank_debug = active_bank;

    // Pipeline validity: capture plus four registered internal transfers.  The
    // decision is published when v4 is consumed, exactly input_cycle + 5.
    reg v0, v1, v2, v3, v4;
    reg [9:0] code_s0;
    reg [1:0] x_s0, z_s0;
    reg phase_s0;
    reg [7:0] ood_s0;
    reg [15:0] age_s0;
    reg reset_ack_s0, observation_valid_s0, deadline_ok_s0, crc_ok_s0;
    reg bank_s0, bank_trusted_s0;
    reg [15:0] version_s0;

    reg [8:0] address_s1;
    reg [1:0] fraction_s1;
    reg [1:0] x_s1, z_s1;
    reg phase_s1;
    reg [7:0] ood_s1;
    reg [15:0] age_s1;
    reg reset_ack_s1, observation_valid_s1, deadline_ok_s1, crc_ok_s1;
    reg bank_s1, bank_trusted_s1;
    reg [15:0] version_s1;

    reg [21:0] bank0_x_y0_q, bank0_x_y1_q;
    reg [21:0] bank0_z_y0_q, bank0_z_y1_q;
    reg [21:0] bank1_x_y0_q, bank1_x_y1_q;
    reg [21:0] bank1_z_y0_q, bank1_z_y1_q;
    reg [8:0] address_s2;
    reg [1:0] fraction_s2;
    reg [1:0] x_s2, z_s2;
    reg phase_s2;
    reg [7:0] ood_s2;
    reg [15:0] age_s2;
    reg reset_ack_s2, observation_valid_s2, deadline_ok_s2, crc_ok_s2;
    reg bank_s2, bank_trusted_s2;
    reg [15:0] version_s2;

    reg signed [21:0] y0_s3;
    reg signed [27:0] product_s3;
    reg [8:0] address_s3;
    reg [1:0] x_s3, z_s3;
    reg phase_s3;
    reg [7:0] ood_s3;
    reg [15:0] age_s3;
    reg reset_ack_s3, observation_valid_s3, deadline_ok_s3, crc_ok_s3;
    reg bank_trusted_s3;
    reg [15:0] version_s3;

    reg signed [25:0] interpolation_s4;
    reg [8:0] address_s4;
    reg [1:0] x_s4, z_s4;
    reg phase_s4;
    reg [7:0] ood_s4;
    reg [15:0] age_s4;
    reg reset_ack_s4, observation_valid_s4, deadline_ok_s4, crc_ok_s4;
    reg bank_trusted_s4;
    reg [15:0] version_s4;

    wire [41:0] input_payload = in_word[41:0];
    wire input_crc_ok = (crc16_42(input_payload) == in_word[57:42]);
    wire input_reserved_observation =
        (input_payload[11:10] == 2'b11) || (input_payload[13:12] == 2'b11);
    wire requested_bank_trusted = commit_bank ? bank1_trusted : bank0_trusted;
    wire selected_commit = commit_valid && safe_boundary &&
                           requested_bank_trusted &&
                           (commit_bank != active_bank) &&
                           (active_version != 16'hffff) &&
                           (commit_version == (active_version + 16'd1));
    wire cfg_write_allowed = cfg_we && (cfg_address <= 9'd256) &&
                             (cfg_bank != active_bank) && !selected_commit;

    wire [21:0] selected_y0_s2 = !bank_s2 ?
        (phase_s2 ? bank0_z_y0_q : bank0_x_y0_q) :
        (phase_s2 ? bank1_z_y0_q : bank1_x_y0_q);
    wire [21:0] selected_y1_s2 = !bank_s2 ?
        (phase_s2 ? bank0_z_y1_q : bank0_x_y1_q) :
        (phase_s2 ? bank1_z_y1_q : bank1_x_y1_q);
    wire signed [22:0] difference_s2 =
        $signed({selected_y1_s2[21], selected_y1_s2}) -
        $signed({selected_y0_s2[21], selected_y0_s2});
    wire signed [4:0] numerator_s2 = $signed({1'b0, fraction_s2, 1'b1});
    wire [21:0] llr_final_bits = saturate_llr(interpolation_s4);
    wire signed [21:0] llr_final_signed = llr_final_bits;

    // Event/fallback state mirrors ExperimentalEventFSM and
    // ConservativeFallbackController for the on-device observable flags.
    reg [2:0] mode;
    reg [2:0] x_e_run, z_e_run, leakage_run, leakage_clean_run;
    reg [2:0] health_good_run, reset_wait_run;
    reg pauli_frame_x, pauli_frame_z;
    reg [7:0] phase_frame_x, phase_frame_z;
    reg [15:0] trusted_active_version;
    reg [2:0] health_status;
    reg [7:0] fault_run, good_run, fault_cycle_count, leakage_cycle_count;
    reg [7:0] fault_counts [0:13];
    reg [13:0] last_fault_mask;

    wire leakage_observed_s4 = (x_s4 == 2'd2) || (z_s4 == 2'd2);
    wire [13:0] event_fault_mask = {
        leakage_observed_s4,
        (reset_ack_s4 && (mode != MODE_RESET_REQUEST)),
        1'b0, // map alignment/action is construction-checked
        1'b0, // map decision is present whenever v4 is consumed
        !deadline_ok_s4,
        (age_s4 > MAX_PARAMETER_AGE_CYCLES),
        (version_s4 < trusted_active_version),
        1'b0, // decision version is the latched version
        (version_s4 > MAX_TRUSTED_BANK_VERSION),
        !bank_trusted_s4,
        !bank_trusted_s4,
        !crc_ok_s4,
        (ood_s4 > OOD_THRESHOLD_CODE),
        !observation_valid_s4
    };
    wire event_blocking = |event_fault_mask[12:0];
    wire event_map_accepted = !event_blocking;
    wire event_flip = llr_final_signed[21];
    wire [15:0] event_trusted_version = event_map_accepted ? version_s4 :
                                                           trusted_active_version;

    wire [2:0] next_x_e_run = event_blocking ? 3'd0 :
        ((x_s4 == 2'd1) ? sat3_inc(x_e_run) : 3'd0);
    wire [2:0] next_z_e_run = event_blocking ? 3'd0 :
        ((z_s4 == 2'd1) ? sat3_inc(z_e_run) : 3'd0);
    wire [2:0] next_leakage_run = event_blocking ? 3'd0 :
        (leakage_observed_s4 ? sat3_inc(leakage_run) : 3'd0);
    wire [2:0] next_leakage_clean_run = event_blocking ? 3'd0 :
        (leakage_observed_s4 ? 3'd0 : sat3_inc(leakage_clean_run));
    wire [2:0] next_health_good_run = event_blocking ? 3'd0 :
        sat3_inc(health_good_run);
    wire [2:0] next_reset_wait_run =
        ((mode == MODE_RESET_REQUEST) && !reset_ack_s4) ? sat3_inc(reset_wait_run) : 3'd0;

    reg [2:0] event_mode;
    always @* begin
        if (event_blocking)
            event_mode = MODE_FALLBACK;
        else if (reset_ack_s4)
            event_mode = (mode == MODE_RESET_REQUEST) ? MODE_HOLD : MODE_FALLBACK;
        else if (mode == MODE_RESET_REQUEST)
            event_mode = MODE_RESET_REQUEST;
        else if (leakage_observed_s4)
            event_mode = (next_leakage_run >= 3'd2) ? MODE_RESET_REQUEST : MODE_HOLD;
        else if ((mode == MODE_HOLD) && (next_leakage_clean_run < 3'd2))
            event_mode = MODE_HOLD;
        else if ((mode == MODE_FALLBACK) && (next_health_good_run < 3'd2))
            event_mode = MODE_FALLBACK;
        else if ((next_x_e_run >= 3'd2) && (next_z_e_run >= 3'd2))
            event_mode = phase_s4 ? MODE_Z_RECOVERY : MODE_X_RECOVERY;
        else if (next_x_e_run >= 3'd2)
            event_mode = MODE_X_RECOVERY;
        else if (next_z_e_run >= 3'd2)
            event_mode = MODE_Z_RECOVERY;
        else
            event_mode = MODE_NORMAL;
    end

    wire event_inhibited = (event_mode == MODE_HOLD) ||
                            (event_mode == MODE_RESET_REQUEST) ||
                            (event_mode == MODE_FALLBACK);
    wire event_apply_map = event_map_accepted && !event_inhibited && event_flip;
    wire event_delta_x = event_apply_map && !phase_s4;
    wire event_delta_z = event_apply_map && phase_s4;
    wire event_pauli_x = pauli_frame_x ^ event_delta_x;
    wire event_pauli_z = pauli_frame_z ^ event_delta_z;
    wire [7:0] event_phase_x = phase_frame_x + (event_delta_x ? 8'h80 : 8'h00);
    wire [7:0] event_phase_z = phase_frame_z + (event_delta_z ? 8'h80 : 8'h00);
    wire [1:0] event_action_code =
        (!event_map_accepted || !event_flip) ? 2'd0 : (phase_s4 ? 2'd2 : 2'd1);

    reg [2:0] event_health_status;
    always @* begin
        if (event_mode == MODE_RESET_REQUEST)
            event_health_status = HEALTH_RESET_REQUIRED;
        else if (event_blocking)
            event_health_status = HEALTH_FALLBACK;
        else if (event_mode == MODE_FALLBACK)
            event_health_status = HEALTH_RECOVERING;
        else if ((event_fault_mask != 14'd0) || (event_mode == MODE_HOLD))
            event_health_status = HEALTH_DEGRADED;
        else
            event_health_status = HEALTH_HEALTHY;
    end

    reg [101:0] pending_output_payload;
    reg [101:0] output_payload;
    reg [215:0] state_payload;
    integer state_index;

    always @* begin
        state_payload = 216'd0;
        state_payload[2:0] = mode;
        state_payload[5:3] = x_e_run;
        state_payload[8:6] = z_e_run;
        state_payload[11:9] = leakage_run;
        state_payload[14:12] = leakage_clean_run;
        state_payload[17:15] = health_good_run;
        state_payload[20:18] = reset_wait_run;
        state_payload[21] = pauli_frame_x;
        state_payload[22] = pauli_frame_z;
        state_payload[30:23] = phase_frame_x;
        state_payload[38:31] = phase_frame_z;
        state_payload[54:39] = trusted_active_version;
        state_payload[57:55] = health_status;
        state_payload[65:58] = fault_run;
        state_payload[73:66] = good_run;
        state_payload[81:74] = fault_cycle_count;
        state_payload[89:82] = leakage_cycle_count;
        for (state_index = 0; state_index < 14; state_index = state_index + 1)
            state_payload[90 + state_index * 8 +: 8] = fault_counts[state_index];
        state_payload[215:202] = last_fault_mask;
    end

    assign out_word = {crc16_102(output_payload), output_payload};
    assign state_word = {crc16_216(state_payload), state_payload};

    integer reset_index;
    integer count_index;
    always @(posedge clk) begin
        if (!reset_n) begin
            active_bank <= 1'b0;
            active_version <= 16'd0;
            commit_ack <= 1'b0;
            v0 <= 1'b0;
            v1 <= 1'b0;
            v2 <= 1'b0;
            v3 <= 1'b0;
            v4 <= 1'b0;
            map_valid_debug <= 1'b0;
            map_llr_debug <= 22'd0;
            map_address_debug <= 9'd0;
            mode <= MODE_NORMAL;
            x_e_run <= 3'd0;
            z_e_run <= 3'd0;
            leakage_run <= 3'd0;
            leakage_clean_run <= 3'd0;
            health_good_run <= 3'd0;
            reset_wait_run <= 3'd0;
            pauli_frame_x <= 1'b0;
            pauli_frame_z <= 1'b0;
            phase_frame_x <= 8'd0;
            phase_frame_z <= 8'd0;
            trusted_active_version <= 16'd0;
            health_status <= HEALTH_HEALTHY;
            fault_run <= 8'd0;
            good_run <= 8'd0;
            fault_cycle_count <= 8'd0;
            leakage_cycle_count <= 8'd0;
            last_fault_mask <= 14'd0;
            pending_output_payload <= 102'd0;
            output_payload <= 102'd0;
            for (reset_index = 0; reset_index < 14; reset_index = reset_index + 1)
                fault_counts[reset_index] <= 8'd0;
        end else begin
            commit_ack <= selected_commit;
            if (selected_commit) begin
                active_bank <= commit_bank;
                active_version <= commit_version;
            end

            if (cfg_write_allowed) begin
                if (!cfg_bank && !cfg_phase) begin
                    bank0_x_y0[cfg_address] <= cfg_data;
                    bank0_x_y1[cfg_address] <= cfg_data;
                end else if (!cfg_bank && cfg_phase) begin
                    bank0_z_y0[cfg_address] <= cfg_data;
                    bank0_z_y1[cfg_address] <= cfg_data;
                end else if (cfg_bank && !cfg_phase) begin
                    bank1_x_y0[cfg_address] <= cfg_data;
                    bank1_x_y1[cfg_address] <= cfg_data;
                end else begin
                    bank1_z_y0[cfg_address] <= cfg_data;
                    bank1_z_y1[cfg_address] <= cfg_data;
                end
            end

            // Output register is one cycle after the MAP/FSM decision.
            output_payload <= pending_output_payload;
            pending_output_payload <= 102'd0;
            map_valid_debug <= v4;

            v0 <= in_valid;
            v1 <= v0;
            v2 <= v1;
            v3 <= v2;
            v4 <= v3;

            if (in_valid) begin
                code_s0 <= input_payload[9:0];
                x_s0 <= input_payload[11:10];
                z_s0 <= input_payload[13:12];
                phase_s0 <= input_payload[14];
                ood_s0 <= input_payload[22:15];
                age_s0 <= input_payload[38:23];
                reset_ack_s0 <= input_payload[39];
                observation_valid_s0 <= input_payload[40] && !input_reserved_observation;
                deadline_ok_s0 <= input_payload[41];
                crc_ok_s0 <= input_crc_ok;
                bank_s0 <= selected_commit ? commit_bank : active_bank;
                bank_trusted_s0 <= selected_commit ?
                    (commit_bank ? bank1_trusted : bank0_trusted) :
                    (active_bank ? bank1_trusted : bank0_trusted);
                version_s0 <= selected_commit ? commit_version : active_version;
            end

            if (v0) begin
                address_s1 <= code_s0[9:2];
                fraction_s1 <= code_s0[1:0];
                x_s1 <= x_s0;
                z_s1 <= z_s0;
                phase_s1 <= phase_s0;
                ood_s1 <= ood_s0;
                age_s1 <= age_s0;
                reset_ack_s1 <= reset_ack_s0;
                observation_valid_s1 <= observation_valid_s0;
                deadline_ok_s1 <= deadline_ok_s0;
                crc_ok_s1 <= crc_ok_s0;
                bank_s1 <= bank_s0;
                bank_trusted_s1 <= bank_trusted_s0;
                version_s1 <= version_s0;
            end

            if (v1) begin
                // Direct registered reads let Yosys merge each output FF into
                // a legal synchronous Gowin BSRAM read port.  Bank/phase
                // selection occurs only after the registered memory outputs.
                bank0_x_y0_q <= bank0_x_y0[address_s1];
                bank0_x_y1_q <= bank0_x_y1[address_s1 + 9'd1];
                bank0_z_y0_q <= bank0_z_y0[address_s1];
                bank0_z_y1_q <= bank0_z_y1[address_s1 + 9'd1];
                bank1_x_y0_q <= bank1_x_y0[address_s1];
                bank1_x_y1_q <= bank1_x_y1[address_s1 + 9'd1];
                bank1_z_y0_q <= bank1_z_y0[address_s1];
                bank1_z_y1_q <= bank1_z_y1[address_s1 + 9'd1];
                address_s2 <= address_s1;
                fraction_s2 <= fraction_s1;
                x_s2 <= x_s1;
                z_s2 <= z_s1;
                phase_s2 <= phase_s1;
                ood_s2 <= ood_s1;
                age_s2 <= age_s1;
                reset_ack_s2 <= reset_ack_s1;
                observation_valid_s2 <= observation_valid_s1;
                deadline_ok_s2 <= deadline_ok_s1;
                crc_ok_s2 <= crc_ok_s1;
                bank_s2 <= bank_s1;
                bank_trusted_s2 <= bank_trusted_s1;
                version_s2 <= version_s1;
            end

            if (v2) begin
                y0_s3 <= selected_y0_s2;
                product_s3 <= difference_s2 * numerator_s2;
                address_s3 <= address_s2;
                x_s3 <= x_s2;
                z_s3 <= z_s2;
                phase_s3 <= phase_s2;
                ood_s3 <= ood_s2;
                age_s3 <= age_s2;
                reset_ack_s3 <= reset_ack_s2;
                observation_valid_s3 <= observation_valid_s2;
                deadline_ok_s3 <= deadline_ok_s2;
                crc_ok_s3 <= crc_ok_s2;
                bank_trusted_s3 <= bank_trusted_s2;
                version_s3 <= version_s2;
            end

            if (v3) begin
                interpolation_s4 <= $signed({{4{y0_s3[21]}}, y0_s3}) +
                                    $signed(round_shift3_ties_even(product_s3));
                address_s4 <= address_s3;
                x_s4 <= x_s3;
                z_s4 <= z_s3;
                phase_s4 <= phase_s3;
                ood_s4 <= ood_s3;
                age_s4 <= age_s3;
                reset_ack_s4 <= reset_ack_s3;
                observation_valid_s4 <= observation_valid_s3;
                deadline_ok_s4 <= deadline_ok_s3;
                crc_ok_s4 <= crc_ok_s3;
                bank_trusted_s4 <= bank_trusted_s3;
                version_s4 <= version_s3;
            end

            if (v4) begin
                map_llr_debug <= llr_final_bits;
                map_address_debug <= address_s4;

                mode <= event_mode;
                x_e_run <= next_x_e_run;
                z_e_run <= next_z_e_run;
                leakage_run <= next_leakage_run;
                leakage_clean_run <= next_leakage_clean_run;
                health_good_run <= next_health_good_run;
                reset_wait_run <= next_reset_wait_run;
                pauli_frame_x <= event_pauli_x;
                pauli_frame_z <= event_pauli_z;
                phase_frame_x <= event_phase_x;
                phase_frame_z <= event_phase_z;
                trusted_active_version <= event_trusted_version;
                health_status <= event_health_status;
                fault_run <= event_blocking ? sat8_inc(fault_run) : 8'd0;
                good_run <= (event_fault_mask == 14'd0) ? sat8_inc(good_run) : 8'd0;
                fault_cycle_count <= event_blocking ? sat8_inc(fault_cycle_count) :
                                                     fault_cycle_count;
                leakage_cycle_count <= event_fault_mask[13] ?
                    sat8_inc(leakage_cycle_count) : leakage_cycle_count;
                last_fault_mask <= event_fault_mask;
                for (count_index = 0; count_index < 14; count_index = count_index + 1)
                    if (event_fault_mask[count_index])
                        fault_counts[count_index] <= sat8_inc(fault_counts[count_index]);

                pending_output_payload[0] <= 1'b1;
                pending_output_payload[3:1] <= event_mode;
                pending_output_payload[4] <= !event_inhibited;
                pending_output_payload[5] <= (event_mode == MODE_RESET_REQUEST);
                pending_output_payload[6] <= event_map_accepted && event_inhibited && event_flip;
                pending_output_payload[8:7] <= event_action_code;
                pending_output_payload[9] <= event_delta_x;
                pending_output_payload[10] <= event_delta_z;
                pending_output_payload[11] <= event_pauli_x;
                pending_output_payload[12] <= event_pauli_z;
                pending_output_payload[20:13] <= event_phase_x;
                pending_output_payload[28:21] <= event_phase_z;
                pending_output_payload[31:29] <= next_x_e_run;
                pending_output_payload[34:32] <= next_z_e_run;
                pending_output_payload[37:35] <= next_leakage_run;
                pending_output_payload[40:38] <= next_leakage_clean_run;
                pending_output_payload[43:41] <= next_health_good_run;
                pending_output_payload[46:44] <= next_reset_wait_run;
                pending_output_payload[49:47] <= event_health_status;
                pending_output_payload[63:50] <= event_fault_mask;
                pending_output_payload[79:64] <= event_trusted_version;
                pending_output_payload[101:80] <= llr_final_bits;
            end
        end
    end

endmodule
