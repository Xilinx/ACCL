`timescale 1 ps / 1 ps

import eci_cmd_defs::*;
import block_types::*;

import lynxTypes::*;

module reorder_buffer_wr #(
    parameter integer           N_THREADS = 32,
    parameter integer           N_BURSTED = 2
) (
    input  logic                aclk,
    input  logic                aresetn,

    // Input
    input  logic [ECI_ADDR_WIDTH-1:0]         axi_in_awaddr,
    input  logic [7:0]          axi_in_awlen,
    output logic                axi_in_awready,
    input  logic                axi_in_awvalid,
    
    output logic  [1:0]         axi_in_bresp,
    input  logic                axi_in_bready,
    output logic                axi_in_bvalid,

    // Output
    output logic [ECI_ADDR_WIDTH-1:0]         axi_out_awaddr,
    output logic [4:0]          axi_out_awid,
    output logic [7:0]          axi_out_awlen,
    input  logic                axi_out_awready,
    output logic                axi_out_awvalid,
    
    input  logic [4:0]          axi_out_bid,
    input  logic [1:0]          axi_out_bresp,
    output logic                axi_out_bready,
    input  logic                axi_out_bvalid
);

// ----------------------------------------------------------------------

localparam integer N_THREADS_BITS = $clog2(N_THREADS);


// ----------------------------------------------------------------------

// Threads
logic [N_THREADS-1:0] threads_C;
logic [N_THREADS-1:0] valid_C;
logic bvalid_C;
logic [4:0] bid_C;

// Pointers
logic [N_THREADS_BITS-1:0] head_C;
logic [N_THREADS_BITS-1:0] tail_C;

// Internal
logic issue_possible;

logic wr_send;
logic wr_recv;

logic stall;

logic [4:0] b_addr;
logic [1:0] b_data;
/*
ila_reorder_buffer_wr inst_ila_reorder_buffer_wr (
    .clk(aclk),
    .probe0(threads_C), // 32
    .probe1(bvalid_C), 
    .probe2(bid_C), // 5
    .probe3(stall), 
    .probe4(head_C), // 5
    .probe5(tail_C), // 5
    .probe6(axi_in_awvalid),
    .probe7(axi_in_awready),
    .probe8(issue_possible),
    .probe9(valid_C), // 32
    .probe10(b_addr), // 5
    .probe11(wr_send),
    .probe12(wr_recv)
);
*/
// -- REG
always_ff @( posedge aclk ) begin : REG_PROC
    if(~aresetn) begin
        threads_C <= 0;
        valid_C <= 0;
        head_C <= 0;
        tail_C <= 0;
        
        bvalid_C <= 1'b0;
        bid_C <= 'X;
    end
    else begin
        bvalid_C <= 1'b0;

        // Send
        if(wr_send) begin
            head_C <= head_C + (axi_in_awlen + 5'd1);
            for(logic[4:0] i = 0; i < N_BURSTED; i++) begin
                if(axi_in_awlen >= i) begin
                    threads_C[head_C + i] <= 1'b1;
                end
            end
        end

        // Receive
        if(wr_recv) begin
            valid_C[axi_out_bid] <= 1'b1;
        end

        // Tail
        if(~stall) begin
            if(valid_C[tail_C] == 1'b1) begin
                threads_C[tail_C] <= 1'b0;
                valid_C[tail_C] <= 1'b0;
                tail_C <= tail_C + 1;

                bvalid_C <= 1'b1;
                bid_C <= tail_C;
            end
        end
        else begin
            bvalid_C <= bvalid_C;
        end

    end
end

// -- DP - issuing
always_comb begin
    issue_possible = 1'b1;
    
    for(logic[4:0] i = 0; i < N_BURSTED; i++) begin
        if(axi_in_awlen >= i) begin
            if(threads_C[head_C + i] == 1'b1) begin
                issue_possible = 1'b0;
            end
        end
    end 
end

// -- DP - read send, drive handshake
always_comb begin
    wr_send = 1'b0;
    axi_in_awready = 1'b0;
    axi_out_awvalid = 1'b0;
    axi_out_awid = head_C;

    // Read
    if(axi_in_awvalid) begin
        if(axi_out_awready && issue_possible) begin
            wr_send = 1'b1;
            axi_in_awready = 1'b1;
            axi_out_awvalid = 1'b1;
        end
    end 
end

// Responses

// -- DP - reponse hshake (axi_out resonse)
always_comb begin
    // Axi out (A port)
    axi_out_bready = 1'b1;
    wr_recv = axi_out_bvalid;

    // Axi in (B port)
    stall = ~axi_in_bready;
    axi_in_bvalid = bvalid_C;
    axi_in_bresp = b_data;
    b_addr = stall ? bid_C : tail_C;
end

// Reorder buffer
ram_tp_nc #(
    .ADDR_BITS(5),
    .DATA_BITS(2)
) inst_reorder_buffer_wr (
    .clk(aclk),
    .a_en(1'b1),
    .a_we(axi_out_bvalid),
    .a_addr(axi_out_bid),
    .a_data_in(axi_out_bresp),
    .a_data_out(),
    .b_en(1'b1),
    .b_addr(b_addr),
    .b_data_out(b_data)
);

// Passthrough
assign axi_out_awaddr 	    = axi_in_awaddr;		
assign axi_out_awlen		= axi_in_awlen;	

endmodule
