// T6.25.2 arbitrary-state one-step refinement for the actual fast-path core
// commit handshake.  The data pipeline is idle so synthesis can remove the
// LUT datapath; the checked active-bank/version/ack logic is unchanged.

module gkp_fast_path_atomic_commit_formal (
    input wire clk
);
    (* anyseq *) reg reset_n;
    (* anyseq *) reg safe_boundary;
    (* anyseq *) reg commit_valid;
    (* anyseq *) reg commit_bank;
    (* anyseq *) reg [15:0] commit_version;
    (* anyseq *) reg bank0_trusted;
    (* anyseq *) reg bank1_trusted;

    reg f_past_valid;
    initial f_past_valid = 1'b0;
    always @(posedge clk)
        f_past_valid <= 1'b1;

    wire commit_ack;
    wire [15:0] active_version;
    wire active_bank;
    wire [117:0] out_word;
    wire [231:0] state_word;
    wire map_valid;
    wire [21:0] map_llr;
    wire [8:0] map_address;
    wire expected_accept = commit_valid && safe_boundary &&
                           (commit_bank ? bank1_trusted : bank0_trusted) &&
                           commit_bank != active_bank &&
                           active_version != 16'hffff &&
                           commit_version == active_version + 16'd1;

    gkp_fast_path_core #(
        .MAX_TRUSTED_BANK_VERSION(16'hffff)
    ) core (
        .clk(clk), .reset_n(reset_n), .in_valid(1'b0), .in_word(58'd0),
        .safe_boundary(safe_boundary), .commit_valid(commit_valid),
        .commit_bank(commit_bank), .commit_version(commit_version),
        .commit_ack(commit_ack), .cfg_we(1'b0), .cfg_bank(1'b0),
        .cfg_phase(1'b0), .cfg_address(9'd0), .cfg_data(22'd0),
        .bank0_trusted(bank0_trusted), .bank1_trusted(bank1_trusted),
        .out_word(out_word), .state_word(state_word),
        .map_valid_debug(map_valid), .map_llr_debug(map_llr),
        .map_address_debug(map_address), .active_version_debug(active_version),
        .active_bank_debug(active_bank)
    );

    always @(posedge clk) begin
        if (f_past_valid && reset_n && $past(reset_n)) begin
            assert(commit_ack == $past(expected_accept));
            if ($past(expected_accept)) begin
                assert(active_bank == $past(commit_bank));
                assert(active_version == $past(active_version) + 16'd1);
            end else begin
                assert(active_bank == $past(active_bank));
                assert(active_version == $past(active_version));
            end
            if ($past(active_version == 16'hffff)) begin
                assert(!commit_ack);
                assert(active_version == 16'hffff);
            end
        end
    end
endmodule
