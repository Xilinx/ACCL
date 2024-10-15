`timescale 1ns / 1ps

import lynxTypes::*;

/**
 * @brief  Meta RR arbitration 
 *
 * Round-robin arbitration for generic meta data.
 *
 * @param N_ID         Number of arbitrations
 * @param DATA_BITS     Number of arbitrated data bits
 */
module meta_arbiter #(
    parameter integer                   N_ID = N_REGIONS,
    parameter integer                   N_ID_BITS = N_REGIONS_BITS,
    parameter integer                   DATA_BITS = AXI_DATA_BITS
) (
	input  logic    					aclk,    
	input  logic    					aresetn,

	// User logic
    metaIntf.s                          s_meta [N_ID],
    metaIntf.m                          m_meta,

    output logic [N_ID_BITS-1:0]        id_out
);

// Internal
logic [N_ID-1:0] ready_snk;
logic [N_ID-1:0] valid_snk;
logic [N_ID-1:0][DATA_BITS-1:0] data_snk;

logic ready_src;
logic valid_src;
logic [DATA_BITS-1:0] data_src;

logic [N_ID_BITS-1:0] rr_reg;
logic [N_ID_BITS-1:0] id;

// --------------------------------------------------------------------------------
// IO
// --------------------------------------------------------------------------------
for(genvar i = 0; i < N_ID; i++) begin
    assign valid_snk[i] = s_meta[i].valid;
    assign s_meta[i].ready = ready_snk[i];
    assign data_snk[i] = s_meta[i].data;    
end

assign m_meta.valid = valid_src;
assign ready_src = m_meta.ready;
assign m_meta.data = data_src;

// --------------------------------------------------------------------------------
// RR
// --------------------------------------------------------------------------------
always_ff @(posedge aclk) begin
	if(aresetn == 1'b0) begin
		rr_reg <= 'X;
	end else begin
        if(valid_src & ready_src) begin 
            rr_reg <= rr_reg + 1;
            if(rr_reg >= N_ID-1)
                rr_reg <= 0;
        end
	end
end

// DP
always_comb begin
    ready_snk = 0;
    valid_src = 1'b0;
    id = 0;

    for(int i = 0; i < N_ID; i++) begin
        if(i+rr_reg >= N_ID) begin
            if(valid_snk[i+rr_reg-N_ID]) begin
                valid_src = valid_snk[i+rr_reg-N_ID];
                id = i+rr_reg-N_ID;
                break;
            end
        end
        else begin
            if(valid_snk[i+rr_reg]) begin
                valid_src = valid_snk[i+rr_reg];
                id = i+rr_reg;
                break;
            end
        end
    end

    ready_snk[id] = ready_src;
    data_src = data_snk[id];
end

assign id_out = id;

endmodule