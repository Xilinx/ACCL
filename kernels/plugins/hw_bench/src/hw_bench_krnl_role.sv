// default_nettype of none prevents implicit wire declaration.
`default_nettype none
module hw_bench_krnl_role #(
  parameter integer C_CMDIN_TDATA_WIDTH      = 32,
  parameter integer C_CMDOUT1_TDATA_WIDTH    = 32,
  parameter integer C_CMDOUT2_TDATA_WIDTH    = 32,
  parameter integer C_CMDTIMESTAMP_TDATA_WIDTH = 64,
  parameter integer C_STSIN_TDATA_WIDTH      = 32,
  parameter integer C_STSOUT1_TDATA_WIDTH    = 32,
  parameter integer C_STSOUT2_TDATA_WIDTH    = 32,
  parameter integer C_STSTIMESTAMP_TDATA_WIDTH    = 64
)
(
  // System Signals
  input  wire                                  ap_clk           ,
  input  wire                                  ap_rst_n         ,
  // Pipe (AXI4-Stream host) interface cmdIn
  input  wire                                  cmdIn_tvalid     ,
  output wire                                  cmdIn_tready     ,
  input  wire [C_CMDIN_TDATA_WIDTH-1:0]        cmdIn_tdata      ,
  input  wire [C_CMDIN_TDATA_WIDTH/8-1:0]      cmdIn_tkeep      ,
  input  wire [C_CMDIN_TDATA_WIDTH/8-1:0]      cmdIn_tstrb      ,
  input  wire                                  cmdIn_tlast      ,
  // Pipe (AXI4-Stream host) interface cmdOut1
  output wire                                  cmdOut1_tvalid   ,
  input  wire                                  cmdOut1_tready   ,
  output wire [C_CMDOUT1_TDATA_WIDTH-1:0]      cmdOut1_tdata    ,
  output wire [C_CMDOUT1_TDATA_WIDTH/8-1:0]    cmdOut1_tkeep    ,
  output wire [C_CMDOUT1_TDATA_WIDTH/8-1:0]    cmdOut1_tstrb    ,
  output wire                                  cmdOut1_tlast    ,
  // Pipe (AXI4-Stream host) interface cmdOut2
  output wire                                  cmdOut2_tvalid   ,
  input  wire                                  cmdOut2_tready   ,
  output wire [C_CMDOUT2_TDATA_WIDTH-1:0]      cmdOut2_tdata    ,
  output wire [C_CMDOUT2_TDATA_WIDTH/8-1:0]    cmdOut2_tkeep    ,
  output wire [C_CMDOUT2_TDATA_WIDTH/8-1:0]    cmdOut2_tstrb    ,
  output wire                                  cmdOut2_tlast    ,
  // Pipe (AXI4-Stream host) interface cmdTimestamp
  output wire                                  cmdTimestamp_tvalid,
  input  wire                                  cmdTimestamp_tready,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH-1:0]   cmdTimestamp_tdata ,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH/8-1:0] cmdTimestamp_tkeep ,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH/8-1:0] cmdTimestamp_tstrb ,
  output wire                                  cmdTimestamp_tlast ,
  // Pipe (AXI4-Stream host) interface stsIn
  input  wire                                  stsIn_tvalid     ,
  output wire                                  stsIn_tready     ,
  input  wire [C_STSIN_TDATA_WIDTH-1:0]        stsIn_tdata      ,
  input  wire [C_STSIN_TDATA_WIDTH/8-1:0]      stsIn_tkeep      ,
  input  wire [C_STSIN_TDATA_WIDTH/8-1:0]      stsIn_tstrb      ,
  input  wire                                  stsIn_tlast      ,
  // Pipe (AXI4-Stream host) interface stsOut1
  output wire                                  stsOut1_tvalid   ,
  input  wire                                  stsOut1_tready   ,
  output wire [C_STSOUT1_TDATA_WIDTH-1:0]      stsOut1_tdata    ,
  output wire [C_STSOUT1_TDATA_WIDTH/8-1:0]    stsOut1_tkeep    ,
  output wire [C_STSOUT1_TDATA_WIDTH/8-1:0]    stsOut1_tstrb    ,
  output wire                                  stsOut1_tlast    ,
  // Pipe (AXI4-Stream host) interface stsOut2
  output wire                                  stsOut2_tvalid   ,
  input  wire                                  stsOut2_tready   ,
  output wire [C_STSOUT2_TDATA_WIDTH-1:0]      stsOut2_tdata    ,
  output wire [C_STSOUT2_TDATA_WIDTH/8-1:0]    stsOut2_tkeep    ,
  output wire [C_STSOUT2_TDATA_WIDTH/8-1:0]    stsOut2_tstrb    ,
  output wire                                  stsOut2_tlast    ,
  // Pipe (AXI4-Stream host) interface stsTimestamp
  output wire                                  stsTimestamp_tvalid   ,
  input  wire                                  stsTimestamp_tready   ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH-1:0]      stsTimestamp_tdata    ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH/8-1:0]    stsTimestamp_tkeep    ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH/8-1:0]    stsTimestamp_tstrb    ,
  output wire                                  stsTimestamp_tlast    
);


timeunit 1ps;
timeprecision 1ps;


(* DONT_TOUCH = "yes" *)
reg                                 areset                         = 1'b0;

// Register and invert reset signal.
always @(posedge ap_clk) begin
  areset <= ~ap_rst_n;
end

// Timestamp
logic [63:0] timestamp;
always @(posedge ap_clk) begin
  if (areset) begin
    timestamp <= '0;
  end
  else begin
    timestamp <= timestamp + 1'b1;  
  end
end 

// cmd
assign cmdIn_tready = cmdOut1_tready & cmdOut2_tready & cmdTimestamp_tready;

assign cmdOut1_tvalid = cmdIn_tvalid & cmdOut1_tready & cmdOut2_tready & cmdTimestamp_tready;
assign cmdOut1_tdata = cmdIn_tdata;
assign cmdOut1_tkeep = cmdIn_tkeep;
assign cmdOut1_tlast = cmdIn_tlast;
assign cmdOut1_tstrb = cmdIn_tstrb;

assign cmdOut2_tvalid = cmdIn_tvalid & cmdOut1_tready & cmdOut2_tready & cmdTimestamp_tready;
assign cmdOut2_tdata = cmdIn_tdata;
assign cmdOut2_tkeep = cmdIn_tkeep;
assign cmdOut2_tlast = cmdIn_tlast;
assign cmdOut2_tstrb = cmdIn_tstrb;

assign cmdTimestamp_tvalid = cmdIn_tvalid & cmdOut1_tready & cmdOut2_tready & cmdTimestamp_tready & cmdIn_tlast;
assign cmdTimestamp_tkeep = '1;
assign cmdTimestamp_tdata = timestamp;
assign cmdTimestamp_tstrb = 0;
assign cmdTimestamp_tlast = cmdIn_tvalid & cmdOut1_tready & cmdOut2_tready & cmdTimestamp_tready & cmdIn_tlast;

// sts
assign stsIn_tready = stsOut1_tready & stsOut2_tready & stsTimestamp_tready;

assign stsOut1_tvalid = stsIn_tvalid & stsOut1_tready & stsOut2_tready & stsTimestamp_tready;
assign stsOut1_tdata = stsIn_tdata;
assign stsOut1_tkeep = stsIn_tkeep;
assign stsOut1_tlast = stsIn_tlast;
assign stsOut1_tstrb = stsIn_tstrb;

assign stsOut2_tvalid = stsIn_tvalid & stsOut1_tready & stsOut2_tready & stsTimestamp_tready;
assign stsOut2_tdata = stsIn_tdata;
assign stsOut2_tkeep = stsIn_tkeep;
assign stsOut2_tlast = stsIn_tlast;
assign stsOut2_tstrb = stsIn_tstrb;

assign stsTimestamp_tvalid = stsIn_tvalid & stsOut1_tready & stsOut2_tready & stsTimestamp_tready;
assign stsTimestamp_tkeep = '1;
assign stsTimestamp_tdata = timestamp;
assign stsTimestamp_tstrb = 0;
assign stsTimestamp_tlast = stsIn_tvalid & stsOut1_tready & stsOut2_tready & stsTimestamp_tready;


endmodule : hw_bench_krnl_role
`default_nettype wire
