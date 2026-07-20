// Integrated lower-bound synthesis top for T5.5.4.  The proven fast core and
// the full quantized-GRU storage/MAC walker coexist; this is not a functional
// quantized GRU and not a T6 transport.

module gkp_fast_path_gru_workload_synth_top (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] seed_in,
    output wire [7:0] status_out
);
    wire [7:0] core_status;
    wire gru_busy;
    wire gru_done;
    wire [16:0] macs;
    wire [9:0] biases;
    wire [31:0] gru_signature;
    reg started;

    gkp_fast_path_synth_top core_harness (
        .clk(clk), .reset_n(reset_n), .seed_in(seed_in), .status_out(core_status)
    );

    quantized_gru_workload_kernel gru_lower_bound (
        .clk(clk),
        .reset_n(reset_n),
        .start(!started),
        .activation_seed({seed_in, seed_in, 2'b01}),
        .busy(gru_busy),
        .done(gru_done),
        .weight_macs_completed(macs),
        .biases_consumed(biases),
        .signature(gru_signature)
    );

    always @(posedge clk) begin
        if (!reset_n)
            started <= 1'b0;
        else if (!started)
            started <= 1'b1;
    end

    assign status_out = core_status ^ gru_signature[7:0]
        ^ macs[7:0] ^ {biases[6:0], gru_busy ^ gru_done};
endmodule
