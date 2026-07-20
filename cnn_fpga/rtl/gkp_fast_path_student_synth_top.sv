// T5.5.3 integrated activity top: the T5.5.2 fast path and the real fixed
// four-state student execute in parallel.  This is still not a T6 transport.

module gkp_fast_path_student_synth_top (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] seed_in,
    output wire [7:0] status_out
);
    wire [7:0] core_status;
    reg [15:0] cycle_counter;
    reg [31:0] signature;
    wire student_busy;
    wire student_done;
    wire student_out_valid;
    wire [3:0] student_out_index;
    wire signed [17:0] student_out_value;
    wire [71:0] student_state;
    wire student_start = !student_busy && (cycle_counter[6:0] == 7'd0);

    gkp_fast_path_synth_top core_harness (
        .clk(clk),
        .reset_n(reset_n),
        .seed_in(seed_in),
        .status_out(core_status)
    );

    low_dimensional_student_kernel student (
        .clk(clk),
        .reset_n(reset_n),
        .start(student_start),
        .outcome_e(cycle_counter[7] ^ seed_in[0]),
        .health_ok(1'b1),
        .busy(student_busy),
        .done(student_done),
        .out_valid(student_out_valid),
        .out_index(student_out_index),
        .out_value(student_out_value),
        .state_debug(student_state)
    );

    always @(posedge clk) begin
        if (!reset_n) begin
            cycle_counter <= 16'd0;
            signature <= 32'h9e3779b9;
        end else begin
            cycle_counter <= cycle_counter + 1'b1;
            signature <= {signature[30:0], signature[31] ^ signature[21]}
                ^ {core_status, student_state[7:0], student_out_index,
                   student_out_valid, student_done, student_busy,
                   student_out_value[8:0]};
        end
    end

    assign status_out = core_status ^ signature[7:0];
endmodule
