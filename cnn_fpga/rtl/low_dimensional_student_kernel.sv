// Fixed Q3.14 implementation of the frozen T4.4.3 four-state student.
// One real multiplier is time-multiplexed across the recurrence and 15-output
// affine head.  A healthy update completes in STATE_DIM*(1+15) cycles.

module low_dimensional_student_kernel #(
    parameter integer STATE_DIM = 4,
    parameter integer WORD_BITS = 18,
    parameter integer FRACTIONAL_BITS = 14
) (
    input  wire                         clk,
    input  wire                         reset_n,
    input  wire                         start,
    input  wire                         outcome_e,
    input  wire                         health_ok,
    output reg                          busy,
    output reg                          done,
    output reg                          out_valid,
    output reg  [3:0]                   out_index,
    output reg signed [WORD_BITS-1:0]   out_value,
    output wire [4*WORD_BITS-1:0]       state_debug
);
    localparam integer OUTPUT_COUNT = 15;
    localparam signed [WORD_BITS-1:0] MAX_CODE = (1 <<< (WORD_BITS-1)) - 1;
    localparam signed [WORD_BITS-1:0] MIN_CODE = -(1 <<< (WORD_BITS-1));

    reg signed [WORD_BITS-1:0] initial_mem [0:3];
    reg signed [WORD_BITS-1:0] decay_mem [0:7];
    reg signed [WORD_BITS-1:0] saturation_mem [0:7];
    reg signed [WORD_BITS-1:0] weight_mem [0:59];
    reg signed [WORD_BITS-1:0] bias_mem [0:14];
    reg signed [WORD_BITS-1:0] state [0:3];
    reg                         output_phase;
    reg                         outcome_latched;
    reg [2:0]                   state_index;
    reg [3:0]                   output_cursor;
    reg signed [WORD_BITS-1:0]  accumulator;
    integer                     reset_index;

    initial begin
        if (STATE_DIM < 1 || STATE_DIM > 4)
            $error("STATE_DIM must be in [1,4]");
        if (WORD_BITS != 18 || FRACTIONAL_BITS != 14)
            $error("frozen coefficient memories require signed Q3.14");
        $readmemh("cnn_fpga/rtl/generated/t5_5_3_student_initial_state.mem", initial_mem);
        $readmemh("cnn_fpga/rtl/generated/t5_5_3_student_outcome_decays.mem", decay_mem);
        $readmemh("cnn_fpga/rtl/generated/t5_5_3_student_outcome_saturations.mem", saturation_mem);
        $readmemh("cnn_fpga/rtl/generated/t5_5_3_student_output_weights.mem", weight_mem);
        $readmemh("cnn_fpga/rtl/generated/t5_5_3_student_output_bias.mem", bias_mem);
    end

    function automatic signed [WORD_BITS-1:0] saturate_wide(
        input signed [WORD_BITS:0] value
    );
        begin
            if (value > $signed(MAX_CODE))
                saturate_wide = MAX_CODE;
            else if (value < $signed(MIN_CODE))
                saturate_wide = MIN_CODE;
            else
                saturate_wide = value[WORD_BITS-1:0];
        end
    endfunction

    function automatic signed [WORD_BITS-1:0] round_product_ties_even(
        input signed [2*WORD_BITS-1:0] value
    );
        reg sign_bit;
        reg [2*WORD_BITS-1:0] magnitude;
        reg [2*WORD_BITS-FRACTIONAL_BITS:0] quotient;
        reg [FRACTIONAL_BITS-1:0] remainder;
        reg [2*WORD_BITS-FRACTIONAL_BITS:0] rounded;
        reg signed [2*WORD_BITS-FRACTIONAL_BITS+1:0] signed_rounded;
        begin
            sign_bit = value[2*WORD_BITS-1];
            magnitude = sign_bit ? -value : value;
            quotient = magnitude >> FRACTIONAL_BITS;
            remainder = magnitude[FRACTIONAL_BITS-1:0];
            rounded = quotient;
            if ((remainder > (1 << (FRACTIONAL_BITS-1))) ||
                ((remainder == (1 << (FRACTIONAL_BITS-1))) && quotient[0]))
                rounded = quotient + 1'b1;
            signed_rounded = sign_bit ? -$signed(rounded) : $signed(rounded);
            if (signed_rounded > $signed(MAX_CODE))
                round_product_ties_even = MAX_CODE;
            else if (signed_rounded < $signed(MIN_CODE))
                round_product_ties_even = MIN_CODE;
            else
                round_product_ties_even = signed_rounded[WORD_BITS-1:0];
        end
    endfunction

    function automatic signed [WORD_BITS-1:0] clamp_output(
        input [3:0] index,
        input signed [WORD_BITS:0] value
    );
        reg signed [WORD_BITS-1:0] bound;
        begin
            bound = (index == 4'd14) ? (1 <<< FRACTIONAL_BITS) : (2 <<< FRACTIONAL_BITS);
            if (value > $signed({1'b0, bound}))
                clamp_output = bound;
            else if (value < -$signed({1'b0, bound}))
                clamp_output = -bound;
            else
                clamp_output = saturate_wide(value);
        end
    endfunction

    wire [3:0] coefficient_state_index = {1'b0, state_index};
    wire [3:0] outcome_offset = outcome_latched ? 4 : 0;
    // Async memory reads are evaluated even on an untaken sequential branch.
    // Clamp the terminal next-bias address so CXXRTL and hardware never form
    // bias_mem[15] while completing output 14.
    wire [3:0] next_output_cursor =
        (output_cursor < OUTPUT_COUNT-1) ? (output_cursor + 1'b1) : (OUTPUT_COUNT-1);
    wire [6:0] weight_address = output_cursor * 4 + coefficient_state_index;
    wire signed [WORD_BITS-1:0] selected_saturation =
        saturation_mem[outcome_offset + coefficient_state_index];
    wire signed [WORD_BITS:0] state_difference_wide =
        $signed(state[state_index]) - $signed(selected_saturation);
    wire signed [WORD_BITS-1:0] state_difference = saturate_wide(state_difference_wide);
    wire signed [WORD_BITS-1:0] multiplier_a = output_phase ?
        weight_mem[weight_address] : decay_mem[outcome_offset + coefficient_state_index];
    wire signed [WORD_BITS-1:0] multiplier_b = output_phase ?
        state[state_index] : state_difference;
    wire signed [2*WORD_BITS-1:0] multiplier_product = multiplier_a * multiplier_b;
    wire signed [WORD_BITS-1:0] rounded_product = round_product_ties_even(multiplier_product);
    wire signed [WORD_BITS:0] recurrence_sum =
        $signed(selected_saturation) + $signed(rounded_product);
    wire signed [WORD_BITS:0] output_sum =
        $signed(accumulator) + $signed(rounded_product);

    assign state_debug = {state[3], state[2], state[1], state[0]};

    always @(posedge clk) begin
        done <= 1'b0;
        out_valid <= 1'b0;
        if (!reset_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            out_index <= 4'd0;
            out_value <= {WORD_BITS{1'b0}};
            output_phase <= 1'b0;
            outcome_latched <= 1'b0;
            state_index <= 3'd0;
            output_cursor <= 4'd0;
            accumulator <= {WORD_BITS{1'b0}};
            for (reset_index = 0; reset_index < 4; reset_index = reset_index + 1)
                state[reset_index] <= initial_mem[reset_index];
        end else if (!health_ok) begin
            busy <= 1'b0;
            output_phase <= 1'b0;
            state_index <= 3'd0;
            output_cursor <= 4'd0;
            accumulator <= {WORD_BITS{1'b0}};
            for (reset_index = 0; reset_index < 4; reset_index = reset_index + 1)
                state[reset_index] <= initial_mem[reset_index];
        end else if (!busy) begin
            if (start) begin
                busy <= 1'b1;
                output_phase <= 1'b0;
                outcome_latched <= outcome_e;
                state_index <= 3'd0;
                output_cursor <= 4'd0;
            end
        end else if (!output_phase) begin
            state[state_index] <= saturate_wide(recurrence_sum);
            if (state_index == STATE_DIM-1) begin
                output_phase <= 1'b1;
                state_index <= 3'd0;
                output_cursor <= 4'd0;
                accumulator <= bias_mem[0];
            end else begin
                state_index <= state_index + 1'b1;
            end
        end else begin
            if (state_index == STATE_DIM-1) begin
                out_valid <= 1'b1;
                out_index <= output_cursor;
                out_value <= clamp_output(output_cursor, output_sum);
                state_index <= 3'd0;
                if (output_cursor == OUTPUT_COUNT-1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    output_phase <= 1'b0;
                end else begin
                    output_cursor <= output_cursor + 1'b1;
                    accumulator <= bias_mem[next_output_cursor];
                end
            end else begin
                accumulator <= saturate_wide(output_sum);
                state_index <= state_index + 1'b1;
            end
        end
    end
endmodule
