// This is a generated file. Use and modify at your own risk.
//////////////////////////////////////////////////////////////////////////////// 
// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps
// Top level of the kernel. Do not modify module name, parameters or ports.
module hw_bench #(
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
  //  Note: A minimum subset of AXI4 memory mapped signals are declared.  AXI
  // signals omitted from these interfaces are automatically inferred with the
  // optimal values for Xilinx accleration platforms.  This allows Xilinx AXI4 Interconnects
  // within the system to be optimized by removing logic for AXI4 protocol
  // features that are not necessary. When adapting AXI4 masters within the RTL
  // kernel that have signals not declared below, it is suitable to add the
  // signals to the declarations below to connect them to the AXI4 Master.
  // 
  // List of ommited signals - effect
  // -------------------------------
  // ID - Transaction ID are used for multithreading and out of order
  // transactions.  This increases complexity. This saves logic and increases Fmax
  // in the system when ommited.
  // SIZE - Default value is log2(data width in bytes). Needed for subsize bursts.
  // This saves logic and increases Fmax in the system when ommited.
  // BURST - Default value (0b01) is incremental.  Wrap and fixed bursts are not
  // recommended. This saves logic and increases Fmax in the system when ommited.
  // LOCK - Not supported in AXI4
  // CACHE - Default value (0b0011) allows modifiable transactions. No benefit to
  // changing this.
  // PROT - Has no effect in current acceleration platforms.
  // QOS - Has no effect in current acceleration platforms.
  // REGION - Has no effect in current acceleration platforms.
  // USER - Has no effect in current acceleration platforms.
  // RESP - Not useful in most acceleration platforms.
  // 
  // AXI4-Stream (slave) interface cmdIn
  input  wire                                  cmdIn_tvalid     ,
  output wire                                  cmdIn_tready     ,
  input  wire [C_CMDIN_TDATA_WIDTH-1:0]        cmdIn_tdata      ,
  input  wire [C_CMDIN_TDATA_WIDTH/8-1:0]      cmdIn_tkeep      ,
  input  wire [C_CMDIN_TDATA_WIDTH/8-1:0]      cmdIn_tstrb      ,
  input  wire                                  cmdIn_tlast      ,
  // AXI4-Stream (master) interface cmdOut1
  output wire                                  cmdOut1_tvalid   ,
  input  wire                                  cmdOut1_tready   ,
  output wire [C_CMDOUT1_TDATA_WIDTH-1:0]      cmdOut1_tdata    ,
  output wire [C_CMDOUT1_TDATA_WIDTH/8-1:0]    cmdOut1_tkeep    ,
  output wire [C_CMDOUT1_TDATA_WIDTH/8-1:0]    cmdOut1_tstrb    ,
  output wire                                  cmdOut1_tlast    ,
  // AXI4-Stream (master) interface cmdOut2
  output wire                                  cmdOut2_tvalid   ,
  input  wire                                  cmdOut2_tready   ,
  output wire [C_CMDOUT2_TDATA_WIDTH-1:0]      cmdOut2_tdata    ,
  output wire [C_CMDOUT2_TDATA_WIDTH/8-1:0]    cmdOut2_tkeep    ,
  output wire [C_CMDOUT2_TDATA_WIDTH/8-1:0]    cmdOut2_tstrb    ,
  output wire                                  cmdOut2_tlast    ,
  // AXI4-Stream (master) interface cmdTimestamp
  output wire                                  cmdTimestamp_tvalid,
  input  wire                                  cmdTimestamp_tready,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH-1:0]   cmdTimestamp_tdata ,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH/8-1:0] cmdTimestamp_tkeep ,
  output wire [C_CMDTIMESTAMP_TDATA_WIDTH/8-1:0] cmdTimestamp_tstrb ,
  output wire                                  cmdTimestamp_tlast ,
  // AXI4-Stream (slave) interface stsIn
  input  wire                                  stsIn_tvalid     ,
  output wire                                  stsIn_tready     ,
  input  wire [C_STSIN_TDATA_WIDTH-1:0]        stsIn_tdata      ,
  input  wire [C_STSIN_TDATA_WIDTH/8-1:0]      stsIn_tkeep      ,
  input  wire [C_STSIN_TDATA_WIDTH/8-1:0]      stsIn_tstrb      ,
  input  wire                                  stsIn_tlast      ,
  // AXI4-Stream (master) interface stsOut1
  output wire                                  stsOut1_tvalid   ,
  input  wire                                  stsOut1_tready   ,
  output wire [C_STSOUT1_TDATA_WIDTH-1:0]      stsOut1_tdata    ,
  output wire [C_STSOUT1_TDATA_WIDTH/8-1:0]    stsOut1_tkeep    ,
  output wire [C_STSOUT1_TDATA_WIDTH/8-1:0]    stsOut1_tstrb    ,
  output wire                                  stsOut1_tlast    ,
  // AXI4-Stream (master) interface stsOut2
  output wire                                  stsOut2_tvalid   ,
  input  wire                                  stsOut2_tready   ,
  output wire [C_STSOUT2_TDATA_WIDTH-1:0]      stsOut2_tdata    ,
  output wire [C_STSOUT2_TDATA_WIDTH/8-1:0]    stsOut2_tkeep    ,
  output wire [C_STSOUT2_TDATA_WIDTH/8-1:0]    stsOut2_tstrb    ,
  output wire                                  stsOut2_tlast    ,
  // AXI4-Stream (master) interface stsTimestamp
  output wire                                  stsTimestamp_tvalid   ,
  input  wire                                  stsTimestamp_tready   ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH-1:0]      stsTimestamp_tdata    ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH/8-1:0]    stsTimestamp_tkeep    ,
  output wire [C_STSTIMESTAMP_TDATA_WIDTH/8-1:0]    stsTimestamp_tstrb    ,
  output wire                                  stsTimestamp_tlast    
);

hw_bench_krnl_role #(
  .C_CMDIN_TDATA_WIDTH      ( C_CMDIN_TDATA_WIDTH      ),
  .C_CMDOUT1_TDATA_WIDTH    ( C_CMDOUT1_TDATA_WIDTH    ),
  .C_CMDOUT2_TDATA_WIDTH    ( C_CMDOUT2_TDATA_WIDTH    ),
  .C_CMDTIMESTAMP_TDATA_WIDTH ( C_CMDTIMESTAMP_TDATA_WIDTH ),
  .C_STSIN_TDATA_WIDTH      ( C_STSIN_TDATA_WIDTH      ),
  .C_STSOUT1_TDATA_WIDTH    ( C_STSOUT1_TDATA_WIDTH    ),
  .C_STSOUT2_TDATA_WIDTH    ( C_STSOUT2_TDATA_WIDTH    ),
  .C_STSTIMESTAMP_TDATA_WIDTH    ( C_STSTIMESTAMP_TDATA_WIDTH    )
)
inst_example (
  .ap_clk            ( ap_clk            ),
  .ap_rst_n          ( ap_rst_n          ),
  .cmdIn_tvalid      ( cmdIn_tvalid      ),
  .cmdIn_tready      ( cmdIn_tready      ),
  .cmdIn_tdata       ( cmdIn_tdata       ),
  .cmdIn_tkeep       ( cmdIn_tkeep       ),
  .cmdIn_tstrb       ( cmdIn_tstrb       ),
  .cmdIn_tlast       ( cmdIn_tlast       ),
  .cmdOut1_tvalid    ( cmdOut1_tvalid    ),
  .cmdOut1_tready    ( cmdOut1_tready    ),
  .cmdOut1_tdata     ( cmdOut1_tdata     ),
  .cmdOut1_tkeep     ( cmdOut1_tkeep     ),
  .cmdOut1_tstrb     ( cmdOut1_tstrb     ),
  .cmdOut1_tlast     ( cmdOut1_tlast     ),
  .cmdOut2_tvalid    ( cmdOut2_tvalid    ),
  .cmdOut2_tready    ( cmdOut2_tready    ),
  .cmdOut2_tdata     ( cmdOut2_tdata     ),
  .cmdOut2_tkeep     ( cmdOut2_tkeep     ),
  .cmdOut2_tstrb     ( cmdOut2_tstrb     ),
  .cmdOut2_tlast     ( cmdOut2_tlast     ),
  .cmdTimestamp_tvalid ( cmdTimestamp_tvalid ),
  .cmdTimestamp_tready ( cmdTimestamp_tready ),
  .cmdTimestamp_tdata  ( cmdTimestamp_tdata  ),
  .cmdTimestamp_tkeep  ( cmdTimestamp_tkeep  ),
  .cmdTimestamp_tstrb  ( cmdTimestamp_tstrb  ),
  .cmdTimestamp_tlast  ( cmdTimestamp_tlast  ),
  .stsIn_tvalid      ( stsIn_tvalid      ),
  .stsIn_tready      ( stsIn_tready      ),
  .stsIn_tdata       ( stsIn_tdata       ),
  .stsIn_tkeep       ( stsIn_tkeep       ),
  .stsIn_tstrb       ( stsIn_tstrb       ),
  .stsIn_tlast       ( stsIn_tlast       ),
  .stsOut1_tvalid    ( stsOut1_tvalid    ),
  .stsOut1_tready    ( stsOut1_tready    ),
  .stsOut1_tdata     ( stsOut1_tdata     ),
  .stsOut1_tkeep     ( stsOut1_tkeep     ),
  .stsOut1_tstrb     ( stsOut1_tstrb     ),
  .stsOut1_tlast     ( stsOut1_tlast     ),
  .stsOut2_tvalid    ( stsOut2_tvalid    ),
  .stsOut2_tready    ( stsOut2_tready    ),
  .stsOut2_tdata     ( stsOut2_tdata     ),
  .stsOut2_tkeep     ( stsOut2_tkeep     ),
  .stsOut2_tstrb     ( stsOut2_tstrb     ),
  .stsOut2_tlast     ( stsOut2_tlast     ),
  .stsTimestamp_tvalid    ( stsTimestamp_tvalid    ),
  .stsTimestamp_tready    ( stsTimestamp_tready    ),
  .stsTimestamp_tdata     ( stsTimestamp_tdata     ),
  .stsTimestamp_tkeep     ( stsTimestamp_tkeep     ),
  .stsTimestamp_tstrb     ( stsTimestamp_tstrb     ),
  .stsTimestamp_tlast     ( stsTimestamp_tlast     )
);

endmodule
`default_nettype wire
