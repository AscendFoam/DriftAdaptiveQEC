// Optimistic lower-bound hardware workload for the complete selected GRU.
// It streams every one of the 72,266 quantized weights through one real MAC,
// then reads all 587 biases.  It is intentionally NOT a functional GRU:
// gate dependencies, activation buffers and nonlinearities are omitted, so its
// measured latency/resources are a lower bound and can only disqualify, never
// qualify, the quantized-GRU enhanced route.

module quantized_gru_workload_kernel (
    input  wire               clk,
    input  wire               reset_n,
    input  wire               start,
    input  wire signed [17:0] activation_seed,
    output reg                busy,
    output reg                done,
    output reg [16:0]         weight_macs_completed,
    output reg [9:0]          biases_consumed,
    output reg [31:0]         signature
);
    localparam integer WEIGHT_COUNT = 72266;
    localparam integer BIAS_COUNT = 587;

    (* ram_style = "block" *) reg signed [7:0] weight_mem [0:WEIGHT_COUNT-1];
    (* ram_style = "block" *) reg signed [17:0] bias_mem [0:BIAS_COUNT-1];
    reg signed [7:0] weight_q;
    reg signed [17:0] bias_q;
    reg [16:0] weight_address;
    reg [9:0] bias_address;
    reg weight_valid;
    reg bias_phase;
    reg signed [17:0] activation;
    reg signed [39:0] accumulator;
    wire signed [25:0] product = weight_q * activation;

    initial begin
        $readmemh("cnn_fpga/rtl/generated/t5_5_4_quantized_gru_weights.mem", weight_mem);
        $readmemh("cnn_fpga/rtl/generated/t5_5_4_quantized_gru_biases.mem", bias_mem);
    end

    always @(posedge clk) begin
        weight_q <= weight_mem[weight_address];
        bias_q <= bias_mem[bias_address];
    end

    always @(posedge clk) begin
        done <= 1'b0;
        if (!reset_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            weight_macs_completed <= 17'd0;
            biases_consumed <= 10'd0;
            signature <= 32'h243f6a88;
            weight_address <= 17'd0;
            bias_address <= 10'd0;
            weight_valid <= 1'b0;
            bias_phase <= 1'b0;
            activation <= 18'sh03579;
            accumulator <= 40'sd0;
        end else if (!busy) begin
            if (start) begin
                busy <= 1'b1;
                weight_macs_completed <= 17'd0;
                biases_consumed <= 10'd0;
                weight_address <= 17'd0;
                bias_address <= 10'd0;
                weight_valid <= 1'b0;
                bias_phase <= 1'b0;
                activation <= activation_seed;
                accumulator <= 40'sd0;
            end
        end else if (!bias_phase) begin
            if (weight_valid) begin
                accumulator <= accumulator + product;
                signature <= {signature[30:0], signature[31] ^ signature[21]}
                    ^ {{6{product[25]}}, product} ^ {15'd0, weight_macs_completed};
                activation <= {activation[16:0], activation[17] ^ activation[10] ^ activation[0]};
                weight_macs_completed <= weight_macs_completed + 1'b1;
                if (weight_macs_completed == WEIGHT_COUNT-1) begin
                    bias_phase <= 1'b1;
                    // bias_q already contains bias_mem[0].  Prefetch address 1
                    // for the next synchronous-ROM edge so bias 0 is not
                    // consumed twice and the terminal bias is not skipped.
                    bias_address <= 10'd1;
                    weight_valid <= 1'b0;
                end
            end
            if (!weight_valid) begin
                weight_valid <= 1'b1;
                weight_address <= 17'd1;
            end else if (weight_address < WEIGHT_COUNT-1) begin
                weight_address <= weight_address + 1'b1;
            end
        end else begin
            accumulator <= accumulator + bias_q;
            signature <= {signature[30:0], signature[31] ^ signature[21]}
                ^ accumulator[31:0] ^ {{14{bias_q[17]}}, bias_q};
            biases_consumed <= biases_consumed + 1'b1;
            if (biases_consumed == BIAS_COUNT-1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end else if (biases_consumed < BIAS_COUNT-2) begin
                bias_address <= bias_address + 1'b1;
            end
        end
    end
endmodule
