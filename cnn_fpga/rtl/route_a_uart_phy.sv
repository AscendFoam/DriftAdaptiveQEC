// Board-candidate UART PHY for the Route-A replay transport.
// 8N1, two-flop RX synchronizer, mid-bit sampling, explicit framing pulse.
// The default 27 MHz / 3 Mbaud ratio is exactly nine clocks per bit.

module route_a_uart_rx #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 3000000
) (
    input  wire       clk,
    input  wire       reset_n,
    input  wire       rx_pin,
    output reg  [7:0] byte_data,
    output reg        byte_valid,
    output reg        framing_error,
    output wire       busy
);
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer HALF_BIT = (CLKS_PER_BIT - 1) / 2;
    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1, S_DATA = 2'd2, S_STOP = 2'd3;

    reg rx_meta, rx_sync;
    reg [1:0] state;
    reg [15:0] clock_count;
    reg [2:0] bit_index;
    reg [7:0] shift;
    assign busy = state != S_IDLE;

    always @(posedge clk) begin
        if (!reset_n) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx_pin;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            clock_count <= 16'd0;
            bit_index <= 3'd0;
            shift <= 8'd0;
            byte_data <= 8'd0;
            byte_valid <= 1'b0;
            framing_error <= 1'b0;
        end else begin
            byte_valid <= 1'b0;
            framing_error <= 1'b0;
            case (state)
                S_IDLE: begin
                    clock_count <= 16'd0;
                    bit_index <= 3'd0;
                    if (!rx_sync)
                        state <= S_START;
                end
                S_START: begin
                    if (clock_count == HALF_BIT) begin
                        clock_count <= 16'd0;
                        if (!rx_sync)
                            state <= S_DATA;
                        else
                            state <= S_IDLE; // rejected false start
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                S_DATA: begin
                    if (clock_count == CLKS_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        shift[bit_index] <= rx_sync;
                        if (bit_index == 3'd7) begin
                            bit_index <= 3'd0;
                            state <= S_STOP;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                S_STOP: begin
                    if (clock_count == CLKS_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        state <= S_IDLE;
                        if (rx_sync) begin
                            byte_data <= shift;
                            byte_valid <= 1'b1;
                        end else begin
                            framing_error <= 1'b1;
                        end
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule


module route_a_uart_tx #(
    parameter integer CLK_HZ = 27000000,
    parameter integer BAUD = 3000000
) (
    input  wire       clk,
    input  wire       reset_n,
    input  wire [7:0] byte_data,
    input  wire       byte_valid,
    output wire       byte_ready,
    output wire       tx_pin,
    output wire       busy
);
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam [1:0] S_IDLE = 2'd0, S_START = 2'd1, S_DATA = 2'd2, S_STOP = 2'd3;

    reg [1:0] state;
    reg [15:0] clock_count;
    reg [2:0] bit_index;
    reg [7:0] shift;

    assign byte_ready = state == S_IDLE;
    assign busy = state != S_IDLE;
    assign tx_pin = (state == S_START) ? 1'b0 :
                    (state == S_DATA) ? shift[bit_index] : 1'b1;

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            clock_count <= 16'd0;
            bit_index <= 3'd0;
            shift <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    clock_count <= 16'd0;
                    bit_index <= 3'd0;
                    if (byte_valid) begin
                        shift <= byte_data;
                        state <= S_START;
                    end
                end
                S_START: begin
                    if (clock_count == CLKS_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        state <= S_DATA;
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                S_DATA: begin
                    if (clock_count == CLKS_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        if (bit_index == 3'd7) begin
                            bit_index <= 3'd0;
                            state <= S_STOP;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                S_STOP: begin
                    if (clock_count == CLKS_PER_BIT - 1) begin
                        clock_count <= 16'd0;
                        state <= S_IDLE;
                    end else begin
                        clock_count <= clock_count + 16'd1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
