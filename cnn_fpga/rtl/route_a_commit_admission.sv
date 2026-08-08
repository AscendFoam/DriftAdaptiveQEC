// T6.25.2: combinational admission/arbitration for policy and host commits.
// Policy safety requests have priority.  Host commits are allowed only in the
// OPEN state and when no policy commit is pending.  The bank manager remains
// the sole inactive-target/trust/CAS authority; duplicating target legality in
// this arbiter made the host CAS path unreachable when the policy selector
// changed and immediately raised its own pending request.

module route_a_commit_admission (
    input  wire          policy_commit_valid,
    input  wire          policy_commit_bank,
    input  wire [15:0]   policy_commit_version,
    input  wire          policy_commit_pending,
    input  wire [2:0]    policy_action,
    input  wire          policy_selected_bank,
    input  wire          host_commit_valid,
    input  wire          host_commit_bank,
    input  wire [15:0]   host_expected_active_version,
    input  wire [15:0]   host_new_activation_version,
    output wire          host_commit_blocked,
    output wire          effective_commit_valid,
    output wire          effective_commit_source_policy,
    output wire          effective_commit_bank,
    output wire [15:0]   effective_expected_active_version,
    output wire [15:0]   effective_new_activation_version
);
    localparam [2:0] ACTION_OPEN = 3'd0;

    wire host_commit_allowed = host_commit_valid && !policy_commit_valid &&
                               !policy_commit_pending &&
                               (policy_action == ACTION_OPEN);

    assign host_commit_blocked = host_commit_valid && !host_commit_allowed;
    assign effective_commit_valid = policy_commit_valid || host_commit_allowed;
    assign effective_commit_source_policy = policy_commit_valid;
    assign effective_commit_bank = policy_commit_valid ?
                                   policy_commit_bank : host_commit_bank;
    assign effective_expected_active_version = policy_commit_valid ?
                                               (policy_commit_version - 16'd1) :
                                               host_expected_active_version;
    assign effective_new_activation_version = policy_commit_valid ?
                                              policy_commit_version :
                                              host_new_activation_version;
endmodule
