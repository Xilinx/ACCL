# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *******************************************************************************/

##################################################################
# DESIGN PROCs
##################################################################
# netStackType - UDP or TCP or RDMA - type of POE attachment generated
# enableDMA - 0/1 - enables DMA command/data interfaces, providing support for send/recv from/to memory, and collectives
# enableArithmetic - 0/1 - enables arithmetic, providing support for reduction collectives and combine primitive
# enableCompression - 0/1 - enables compression feature
# enableExtKrnlStream - 0/1 - enables PL stream attachments, providing support for non-memory send/recv
# debugLevel - 0/1/2 - enables DEBUG/TRACE support for the control microblaze
# commitHash - first 24 bits of the commit hash from which we're building
proc create_root_design { netStackType enableDMA enableArithmetic enableCompression enableExtKrnlStream debugLevel {commitHash 0xcafeba} } {

  if { ( $enableDMA == 0 ) && ( $enableExtKrnlStream == 0) } {
      catch {common::send_gid_msg -severity "ERROR" "No data sources and sinks enabled, please enable either DMAs or Streams"}
      return
  }

  if { $debugLevel > 2 } {
      catch {common::send_gid_msg -severity "ERROR" "Debug level can only take values: 0, 1, 2"}
      return
  }

  # Create interface ports
  set s_axi_control [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH {13} \
   CONFIG.ARUSER_WIDTH {0} \
   CONFIG.AWUSER_WIDTH {0} \
   CONFIG.BUSER_WIDTH {0} \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_BRESP {1} \
   CONFIG.HAS_BURST {0} \
   CONFIG.HAS_CACHE {0} \
   CONFIG.HAS_LOCK {0} \
   CONFIG.HAS_PROT {1} \
   CONFIG.HAS_QOS {0} \
   CONFIG.HAS_REGION {0} \
   CONFIG.HAS_RRESP {1} \
   CONFIG.HAS_WSTRB {1} \
   CONFIG.ID_WIDTH {0} \
   CONFIG.MAX_BURST_LENGTH {1} \
   CONFIG.NUM_READ_OUTSTANDING {1} \
   CONFIG.NUM_READ_THREADS {1} \
   CONFIG.NUM_WRITE_OUTSTANDING {1} \
   CONFIG.NUM_WRITE_THREADS {1} \
   CONFIG.PROTOCOL {AXI4LITE} \
   CONFIG.READ_WRITE_MODE {READ_WRITE} \
   CONFIG.RUSER_BITS_PER_BYTE {0} \
   CONFIG.RUSER_WIDTH {0} \
   CONFIG.SUPPORTS_NARROW_BURST {0} \
   CONFIG.WUSER_BITS_PER_BYTE {0} \
   CONFIG.WUSER_WIDTH {0} \
   ] $s_axi_control

  set m_axis_eth_tx_data [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_eth_tx_data ]
  set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_eth_tx_data
  set s_axis_eth_rx_data [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_rx_data ]
  set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_rx_data

  set s_axis_call_req [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_call_req ]
  set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {4} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_call_req
  set m_axis_call_ack [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_call_ack ]
  set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {4} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_call_ack

  # Create ports
  set ap_clk [ create_bd_port -dir I -type clk -freq_hz 250000000 ap_clk ]
  set ap_rst_n [ create_bd_port -dir I -type rst ap_rst_n ]

  set interfaces "s_axi_control:s_axis_call_req:m_axis_call_ack:m_axis_eth_tx_data:s_axis_eth_rx_data"

  # Create instance: axis_switch_0, and set properties
  # We route anything with TDEST > 9 to M9 which will go to external kernels bypassing the segmenter
  set axis_switch_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_0 ]
  set_property -dict [ list \
   CONFIG.DECODER_REG {1} \
   CONFIG.HAS_TSTRB.VALUE_SRC USER \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {10} \
   CONFIG.NUM_SI {8} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH.VALUE_SRC USER \
   CONFIG.ROUTING_MODE {0} \
   CONFIG.TDEST_WIDTH {8} \
   CONFIG.ARB_ON_TLAST {1} \
   CONFIG.ARB_ALGORITHM {3} \
   CONFIG.ARB_ON_MAX_XFERS {0} \
   CONFIG.M09_AXIS_HIGHTDEST {0x000000ff} \
 ] $axis_switch_0

  set control_xbar [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_xbar ]
  set_property -dict [ list CONFIG.NUM_MI {2} ] $control_xbar

  source -notrace ./tcl/control_bd.tcl  
  set idcode [expr {$commitHash<<8 | $debugLevel<<6 | $enableExtKrnlStream<<5 | $enableCompression<<4 | $enableArithmetic<<3 | $enableDMA<<2 | ($netStackType == "RDMA" ? 2 : $netStackType == "TCP" ? 1 : 0) }]
  create_hier_cell_control [current_bd_instance .] control $debugLevel $idcode

  if { $enableDMA == 1 } {
    set m_axis_dma0_s2mm [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma0_s2mm ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma0_s2mm
    set s_axis_dma0_mm2s [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma0_mm2s ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma0_mm2s
    set m_axis_dma1_s2mm [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma1_s2mm ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma1_s2mm
    set s_axis_dma1_mm2s [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma1_mm2s ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma1_mm2s
    set m_axis_dma0_mm2s_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma0_mm2s_cmd ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma0_mm2s_cmd
    set s_axis_dma0_mm2s_sts [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma0_mm2s_sts ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {1} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma0_mm2s_sts
    set m_axis_dma0_s2mm_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma0_s2mm_cmd ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma0_s2mm_cmd
    set s_axis_dma0_s2mm_sts [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma0_s2mm_sts ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {4} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma0_s2mm_sts
    set m_axis_dma1_mm2s_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma1_mm2s_cmd ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma1_mm2s_cmd
    set s_axis_dma1_mm2s_sts [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma1_mm2s_sts ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {1} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma1_mm2s_sts
    set m_axis_dma1_s2mm_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_dma1_s2mm_cmd ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_dma1_s2mm_cmd
    set s_axis_dma1_s2mm_sts [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_dma1_s2mm_sts ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {4} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_dma1_s2mm_sts
    set interfaces "$interfaces:m_axis_dma0_s2mm:s_axis_dma0_mm2s:m_axis_dma1_s2mm:s_axis_dma1_mm2s"
    set interfaces "$interfaces:m_axis_dma0_s2mm_cmd:s_axis_dma0_s2mm_sts:m_axis_dma0_mm2s_cmd:s_axis_dma0_mm2s_sts"
    set interfaces "$interfaces:m_axis_dma1_s2mm_cmd:s_axis_dma1_s2mm_sts:m_axis_dma1_mm2s_cmd:s_axis_dma1_mm2s_sts"
  
    # Segmenters and FIFOs and segmenter command generators for outgoing DMA data
    create_bd_cell -type ip -vlnv xilinx.com:hls:dma2seg_cmd:1.0 dma2segcmd_dmawr_0
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 stream_segmenter_dmawr_0
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 segcmd_fifo_dmawr_0
    set_property -dict [list CONFIG.FIFO_DEPTH {64} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells segcmd_fifo_dmawr_0]
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 data_fifo_dmawr_0
    set_property -dict [list CONFIG.FIFO_DEPTH {64} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells data_fifo_dmawr_0]

    create_bd_cell -type ip -vlnv xilinx.com:hls:dma2seg_cmd:1.0 dma2segcmd_dmawr_1
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 stream_segmenter_dmawr_1
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 segcmd_fifo_dmawr_1
    set_property -dict [list CONFIG.FIFO_DEPTH {64} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells segcmd_fifo_dmawr_1]
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 data_fifo_dmawr_1
    set_property -dict [list CONFIG.FIFO_DEPTH {64} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells data_fifo_dmawr_1]

    # DMA connections
    for {set i 0} {$i < 2} {incr i} {
      connect_bd_intf_net -intf_net dma${i}_rx_cmd [get_bd_intf_pins control/dma${i}_mm2s_cmd] [get_bd_intf_pins m_axis_dma${i}_mm2s_cmd]
      connect_bd_intf_net -intf_net dma${i}_rx_sts [get_bd_intf_pins control/dma${i}_mm2s_sts] [get_bd_intf_pins s_axis_dma${i}_mm2s_sts]
      connect_bd_intf_net -intf_net dma${i}_tx_cmd [get_bd_intf_pins control/dma${i}_s2mm_cmd] [get_bd_intf_pins dma2segcmd_dmawr_${i}/dma_cmd_in]
      connect_bd_intf_net -intf_net dma${i}_tx_sts [get_bd_intf_pins control/dma${i}_s2mm_sts] [get_bd_intf_pins s_axis_dma${i}_s2mm_sts]
      connect_bd_intf_net [get_bd_intf_pins dma2segcmd_dmawr_${i}/dma_cmd_out] [get_bd_intf_pins m_axis_dma${i}_s2mm_cmd]
      connect_bd_intf_net [get_bd_intf_pins dma2segcmd_dmawr_${i}/seg_cmd] [get_bd_intf_pins segcmd_fifo_dmawr_${i}/S_AXIS]
      connect_bd_intf_net [get_bd_intf_pins segcmd_fifo_dmawr_${i}/M_AXIS] [get_bd_intf_pins stream_segmenter_dmawr_${i}/cmd]
      connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M0${i}_AXIS] [get_bd_intf_pins data_fifo_dmawr_${i}/S_AXIS]
      connect_bd_intf_net [get_bd_intf_pins data_fifo_dmawr_${i}/M_AXIS] [get_bd_intf_pins stream_segmenter_dmawr_${i}/in_r]
      connect_bd_intf_net [get_bd_intf_pins stream_segmenter_dmawr_${i}/out_r] [get_bd_intf_pins m_axis_dma${i}_s2mm]
    }

    # Segmenters for DMA to Switch
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_dma0_rd
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_dma1_rd

    connect_bd_intf_net [get_bd_intf_pins s_axis_dma0_mm2s] [get_bd_intf_pins sseg_dma0_rd/in_r]
    connect_bd_intf_net [get_bd_intf_pins s_axis_dma1_mm2s] [get_bd_intf_pins sseg_dma1_rd/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_dma0_rd/out_r] [get_bd_intf_pins axis_switch_0/S00_AXIS]
    connect_bd_intf_net [get_bd_intf_pins sseg_dma1_rd/out_r] [get_bd_intf_pins axis_switch_0/S01_AXIS]
    connect_bd_intf_net [get_bd_intf_pins sseg_dma0_rd/cmd] [get_bd_intf_pins control/dma0_rd_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_dma1_rd/cmd] [get_bd_intf_pins control/dma1_rd_seg_cmd]
  
    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins dma2segcmd_dmawr_0/ap_clk] \
                   [get_bd_pins dma2segcmd_dmawr_1/ap_clk] \
                   [get_bd_pins stream_segmenter_dmawr_0/ap_clk] \
                   [get_bd_pins stream_segmenter_dmawr_1/ap_clk] \
                   [get_bd_pins segcmd_fifo_dmawr_0/s_axis_aclk] \
                   [get_bd_pins segcmd_fifo_dmawr_1/s_axis_aclk] \
                   [get_bd_pins data_fifo_dmawr_0/s_axis_aclk] \
                   [get_bd_pins data_fifo_dmawr_1/s_axis_aclk] \
                   [get_bd_pins sseg_dma0_rd/ap_clk] \
                   [get_bd_pins sseg_dma1_rd/ap_clk]
    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins dma2segcmd_dmawr_0/ap_rst_n] \
                   [get_bd_pins dma2segcmd_dmawr_1/ap_rst_n] \
                   [get_bd_pins stream_segmenter_dmawr_0/ap_rst_n] \
                   [get_bd_pins stream_segmenter_dmawr_1/ap_rst_n] \
                   [get_bd_pins segcmd_fifo_dmawr_0/s_axis_aresetn] \
                   [get_bd_pins segcmd_fifo_dmawr_1/s_axis_aresetn] \
                   [get_bd_pins data_fifo_dmawr_0/s_axis_aresetn] \
                   [get_bd_pins data_fifo_dmawr_1/s_axis_aresetn] \
                   [get_bd_pins sseg_dma0_rd/ap_rst_n] \
                   [get_bd_pins sseg_dma1_rd/ap_rst_n]
  }

  save_bd_design

  # Create address segments

  # Exchange memory, accessible by host and Microblaze at the same offsets
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces s_axi_control] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_0/S_AXI/Mem0] -force

  assign_bd_address -offset 0x00010000 -range 0x00008000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/microblaze_0_local_memory/dlmb_bram_if_cntlr/SLMB/Mem] -force
  assign_bd_address -offset 0x00010000 -range 0x00008000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Instruction] [get_bd_addr_segs control/microblaze_0_local_memory/ilmb_bram_if_cntlr/SLMB/Mem] -force
  assign_bd_address -offset 0x40000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/exchange_mem/axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40001000 -range 0x00001000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/exchange_mem/axi_gpio_1/S_AXI/Reg] -force

  assign_bd_address -offset 0x00050000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/rxbuf_offload/rxbuf_dequeue/s_axi_control/Reg]
  assign_bd_address -offset 0x00060000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/rxbuf_offload/rxbuf_enqueue/s_axi_control/Reg]
  assign_bd_address -offset 0x00070000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs control/rxbuf_offload/rxbuf_seek/s_axi_control/Reg]

  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces control/dma_offload/dma_mover/Data_m_axi_mem] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_bypass/S_AXI/Mem0]
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces control/rxbuf_offload/rxbuf_dequeue/Data_m_axi_mem] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_bypass/S_AXI/Mem0]
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces control/rxbuf_offload/rxbuf_enqueue/Data_m_axi_mem] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_bypass/S_AXI/Mem0]
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces control/rxbuf_offload/rxbuf_seek/Data_m_axi_mem] [get_bd_addr_segs control/exchange_mem/axi_bram_ctrl_bypass/S_AXI/Mem0]

  save_bd_design

  # Create network (de)packetizer
  source -notrace ./tcl/rx_bd.tcl
  source -notrace ./tcl/tx_bd.tcl
  if { $netStackType == "TCP" } {
  
    # TCP interfaces
    set m_axis_eth_read_pkg [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_eth_read_pkg ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_eth_read_pkg

    set m_axis_eth_tx_meta [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_eth_tx_meta ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_eth_tx_meta
  
    set s_axis_eth_notification [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_notification ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {16} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_notification

    set s_axis_eth_rx_meta [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_rx_meta ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {2} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_rx_meta
  
    set s_axis_eth_tx_status [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_tx_status ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {8} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_tx_status
  
    set interfaces "$interfaces:m_axis_eth_read_pkg:m_axis_eth_tx_meta:s_axis_eth_notification:s_axis_eth_rx_meta:s_axis_eth_tx_status"
  
    create_tcp_tx_subsystem [current_bd_instance .] eth_tx_subsystem
    create_tcp_rx_subsystem [current_bd_instance .] eth_rx_subsystem

    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_pktsts] [get_bd_intf_pins control/eth_depacketizer_sts]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/s_axis_pktcmd] [get_bd_intf_pins control/eth_packetizer_cmd]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/m_axis_packetizer_sts] [get_bd_intf_pins control/eth_packetizer_sts]
  
    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_notification] [get_bd_intf_pins control/eth_depacketizer_notif]

    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_rx_data] [get_bd_intf_pins eth_rx_subsystem/s_axis_rx_data]
    connect_bd_intf_net [get_bd_intf_ports m_axis_eth_read_pkg] [get_bd_intf_pins eth_rx_subsystem/m_axis_read_pkg]
    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_rx_meta] [get_bd_intf_pins eth_rx_subsystem/s_axis_rx_meta]
    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_notification] [get_bd_intf_pins eth_rx_subsystem/s_axis_notification]
  
    connect_bd_intf_net [get_bd_intf_ports m_axis_eth_tx_meta] [get_bd_intf_pins eth_tx_subsystem/m_axis_tx_meta]
    connect_bd_intf_net [get_bd_intf_ports m_axis_eth_tx_data] [get_bd_intf_pins eth_tx_subsystem/m_axis_tx_data]
    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_tx_status] [get_bd_intf_pins eth_tx_subsystem/s_axis_tx_status]

    connect_bd_intf_net -intf_net tcp_depacketizer_control [get_bd_intf_pins control_xbar/M01_AXI] [get_bd_intf_pins eth_rx_subsystem/s_axi_control]
    connect_bd_intf_net -intf_net tcp_packetizer_control [get_bd_intf_pins control_xbar/M00_AXI] [get_bd_intf_pins eth_tx_subsystem/s_axi_control]
    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_rx_data] [get_bd_intf_pins axis_switch_0/S02_AXIS]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/s_axis_tx_data] [get_bd_intf_pins axis_switch_0/M02_AXIS]

    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins eth_rx_subsystem/ap_clk] \
                   [get_bd_pins eth_tx_subsystem/ap_clk]

    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins eth_rx_subsystem/ap_rst_n] \
                   [get_bd_pins eth_tx_subsystem/ap_rst_n]

    assign_bd_address -offset 0x00030000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_rx_subsystem/tcp_depacketizer_0/s_axi_control/Reg] -force
    assign_bd_address -offset 0x00040000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_tx_subsystem/tcp_packetizer_0/s_axi_control/Reg] -force

  } elseif { $netStackType == "RDMA" } {
    
    # RDMA interfaces
    set s_axis_eth_notification [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_eth_notification ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TREADY {1} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {8} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_eth_notification
  
    set m_axis_rdma_sq [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rdma_sq ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} ] $m_axis_rdma_sq

    create_rdma_tx_subsystem [current_bd_instance .] eth_tx_subsystem
    create_rdma_rx_subsystem [current_bd_instance .] eth_rx_subsystem

    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_pktsts] [get_bd_intf_pins control/eth_depacketizer_sts]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/s_axis_pktcmd] [get_bd_intf_pins control/eth_packetizer_cmd]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/m_axis_packetizer_sts] [get_bd_intf_pins control/eth_packetizer_sts]

    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_notification] [get_bd_intf_pins control/eth_depacketizer_notif]

    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_rx_data] [get_bd_intf_pins eth_rx_subsystem/s_axis_rx_data]
    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_notification] [get_bd_intf_pins eth_rx_subsystem/s_axis_notification]

    connect_bd_intf_net [get_bd_intf_ports m_axis_eth_tx_data] [get_bd_intf_pins eth_tx_subsystem/m_axis_tx_data]

    connect_bd_intf_net -intf_net rdma_depacketizer_control [get_bd_intf_pins control_xbar/M01_AXI] [get_bd_intf_pins eth_rx_subsystem/s_axi_control]
    connect_bd_intf_net -intf_net rdma_packetizer_control [get_bd_intf_pins control_xbar/M00_AXI] [get_bd_intf_pins eth_tx_subsystem/s_axi_control]
    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_rx_data] [get_bd_intf_pins axis_switch_0/S02_AXIS]
    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/s_axis_tx_data] [get_bd_intf_pins axis_switch_0/M02_AXIS]

    connect_bd_intf_net [get_bd_intf_pins control/m_axis_ub_sq] [get_bd_intf_pins eth_tx_subsystem/s_axis_ub_sq]
    connect_bd_intf_net [get_bd_intf_pins control/s_axis_ub_rq] [get_bd_intf_pins eth_rx_subsystem/m_axis_ub_rq]

    connect_bd_intf_net [get_bd_intf_pins eth_tx_subsystem/m_axis_rdma_sq] [get_bd_intf_pins m_axis_rdma_sq]

    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins eth_rx_subsystem/ap_clk] \
                   [get_bd_pins eth_tx_subsystem/ap_clk]

    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins eth_rx_subsystem/ap_rst_n] \
                   [get_bd_pins eth_tx_subsystem/ap_rst_n]

    assign_bd_address -offset 0x00030000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_rx_subsystem/rdma_depacketizer_0/s_axi_control/Reg] -force
    assign_bd_address -offset 0x00040000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_tx_subsystem/rdma_packetizer_0/s_axi_control/Reg] -force

  } elseif { $netStackType == "UDP" } {

    create_udp_tx_subsystem [current_bd_instance .] eth_tx_subsystem
    create_udp_rx_subsystem [current_bd_instance .] eth_rx_subsystem

    connect_bd_intf_net [get_bd_intf_ports s_axis_eth_rx_data] [get_bd_intf_pins eth_rx_subsystem/s_axis_data]
    connect_bd_intf_net [get_bd_intf_pins control/eth_depacketizer_sts] [get_bd_intf_pins eth_rx_subsystem/m_axis_status]
    connect_bd_intf_net [get_bd_intf_pins control/eth_packetizer_cmd] [get_bd_intf_pins eth_tx_subsystem/s_axis_command]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M02_AXIS] [get_bd_intf_pins eth_tx_subsystem/s_axis_data]
    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_data] [get_bd_intf_pins axis_switch_0/S02_AXIS]
    connect_bd_intf_net [get_bd_intf_ports m_axis_eth_tx_data] [get_bd_intf_pins eth_tx_subsystem/m_axis_data]
    connect_bd_intf_net [get_bd_intf_pins control/eth_packetizer_sts] [get_bd_intf_pins eth_tx_subsystem/m_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins eth_rx_subsystem/m_axis_notification] [get_bd_intf_pins control/eth_depacketizer_notif]
    connect_bd_intf_net -intf_net udp_packetizer_control [get_bd_intf_pins control_xbar/M00_AXI] [get_bd_intf_pins eth_tx_subsystem/s_axi_control]
    connect_bd_intf_net -intf_net udp_depacketizer_control [get_bd_intf_pins control_xbar/M01_AXI] [get_bd_intf_pins eth_rx_subsystem/s_axi_control]

    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins eth_rx_subsystem/ap_clk] \
                   [get_bd_pins eth_tx_subsystem/ap_clk]

    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins eth_rx_subsystem/ap_rst_n] \
                   [get_bd_pins eth_tx_subsystem/ap_rst_n]

    assign_bd_address -offset 0x00030000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_rx_subsystem/udp_depacketizer_0/s_axi_control/Reg] -force
    assign_bd_address -offset 0x00040000 -range 0x00010000 -target_address_space [get_bd_addr_spaces control/microblaze_0/Data] [get_bd_addr_segs eth_tx_subsystem/udp_packetizer_0/s_axi_control/Reg] -force

  } else {
    catch {common::send_gid_msg -severity "ERROR" "Unsupported network stack $netStackType"}
    return
  }

  save_bd_design

  if { $enableArithmetic == 1 } {

    set m_axis_arith_op0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_arith_op0]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {4} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_arith_op0
    set m_axis_arith_op1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_arith_op1]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {4} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_arith_op1
    set s_axis_arith_res [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_arith_res ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_arith_res

    set interfaces "$interfaces:m_axis_arith_op0:m_axis_arith_op1:s_axis_arith_res"

    # Segmenter for result
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_arith_res
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_arith_op0
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_arith_op1

    # Arithmetic-specific connections
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M03_AXIS] [get_bd_intf_pins sseg_arith_op0/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_op0/out_r] [get_bd_intf_ports m_axis_arith_op0]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_op0/cmd] [get_bd_intf_pins control/arith_op0_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M04_AXIS] [get_bd_intf_pins sseg_arith_op1/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_op1/out_r] [get_bd_intf_ports m_axis_arith_op1]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_op1/cmd] [get_bd_intf_pins control/arith_op1_seg_cmd]
    connect_bd_intf_net [get_bd_intf_ports s_axis_arith_res] [get_bd_intf_pins sseg_arith_res/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_res/out_r] [get_bd_intf_pins axis_switch_0/S03_AXIS]
    connect_bd_intf_net [get_bd_intf_pins sseg_arith_res/cmd] [get_bd_intf_pins control/arith_res_seg_cmd]

    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins sseg_arith_res/ap_clk] \
                   [get_bd_pins sseg_arith_op0/ap_clk] \
                   [get_bd_pins sseg_arith_op1/ap_clk] \
                   [get_bd_pins ss_cmd_op_bcast/aclk]

    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins sseg_arith_res/ap_rst_n] \
                   [get_bd_pins sseg_arith_op0/ap_rst_n] \
                   [get_bd_pins sseg_arith_op1/ap_rst_n] \
                   [get_bd_pins ss_cmd_op_bcast/aresetn]
  }

  save_bd_design 

  if { $enableCompression == 1 } {

    set s_axis_compression0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_compression0 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_compression0
  
    set m_axis_compression0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_compression0 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {4} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_compression0
  
    set s_axis_compression1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_compression1 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_compression1
  
    set m_axis_compression1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_compression1 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {4} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_compression1
  
    set s_axis_compression2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_compression2 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {0} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_compression2
  
    set m_axis_compression2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_compression2 ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {4} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_compression2

    set interfaces "$interfaces:s_axis_compression0:m_axis_compression0"
    set interfaces "$interfaces:s_axis_compression1:m_axis_compression1"
    set interfaces "$interfaces:s_axis_compression2:m_axis_compression2"

    # instantiate segmenters
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c0_res
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c1_res
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c2_res

    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c0_op
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c1_op
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_c2_op

    # (de)compression connections
    connect_bd_intf_net [get_bd_intf_ports s_axis_compression0] [get_bd_intf_pins sseg_c0_res/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c0_res/out_r] [get_bd_intf_pins axis_switch_0/S05_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M06_AXIS] [get_bd_intf_pins sseg_c0_op/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c0_op/out_r] [get_bd_intf_ports m_axis_compression0]
 
    connect_bd_intf_net [get_bd_intf_ports s_axis_compression1] [get_bd_intf_pins sseg_c1_res/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c1_res/out_r] [get_bd_intf_pins axis_switch_0/S06_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M07_AXIS] [get_bd_intf_pins sseg_c1_op/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c1_op/out_r] [get_bd_intf_ports m_axis_compression1] 
 
    connect_bd_intf_net [get_bd_intf_ports s_axis_compression2] [get_bd_intf_pins sseg_c2_res/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c2_res/out_r] [get_bd_intf_pins axis_switch_0/S07_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M08_AXIS] [get_bd_intf_pins sseg_c2_op/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_c2_op/out_r] [get_bd_intf_ports m_axis_compression2]

    connect_bd_intf_net [get_bd_intf_pins sseg_c0_op/cmd] [get_bd_intf_pins control/clane0_op_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_c1_op/cmd] [get_bd_intf_pins control/clane1_op_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_c2_op/cmd] [get_bd_intf_pins control/clane2_op_seg_cmd]

    connect_bd_intf_net [get_bd_intf_pins sseg_c0_res/cmd] [get_bd_intf_pins control/clane0_res_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_c1_res/cmd] [get_bd_intf_pins control/clane1_res_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_c2_res/cmd] [get_bd_intf_pins control/clane2_res_seg_cmd] 

    connect_bd_net [get_bd_ports ap_clk] \
                   [get_bd_pins sseg_c0_res/ap_clk] \
                   [get_bd_pins sseg_c1_res/ap_clk] \
                   [get_bd_pins sseg_c2_res/ap_clk] \
                   [get_bd_pins sseg_c0_op/ap_clk] \
                   [get_bd_pins sseg_c1_op/ap_clk] \
                   [get_bd_pins sseg_c2_op/ap_clk]

    connect_bd_net [get_bd_pins control/encore_aresetn] \
                   [get_bd_pins sseg_c0_res/ap_rst_n] \
                   [get_bd_pins sseg_c1_res/ap_rst_n] \
                   [get_bd_pins sseg_c2_res/ap_rst_n] \
                   [get_bd_pins sseg_c0_op/ap_rst_n] \
                   [get_bd_pins sseg_c1_op/ap_rst_n] \
                   [get_bd_pins sseg_c2_op/ap_rst_n]
  } else {
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M06_AXIS] [get_bd_intf_pins axis_switch_0/S05_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M07_AXIS] [get_bd_intf_pins axis_switch_0/S06_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M08_AXIS] [get_bd_intf_pins axis_switch_0/S07_AXIS]
  }

  save_bd_design

  if { $enableExtKrnlStream == 1 } {

    set s_axis_krnl [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_krnl ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $s_axis_krnl
    set m_axis_krnl [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_krnl ]
    set_property -dict [ list CONFIG.FREQ_HZ {250000000} CONFIG.HAS_TKEEP {1} CONFIG.HAS_TLAST {1} CONFIG.HAS_TREADY {1} CONFIG.HAS_TSTRB {0} CONFIG.LAYERED_METADATA {undef} CONFIG.TDATA_NUM_BYTES {64} CONFIG.TDEST_WIDTH {8} CONFIG.TID_WIDTH {0} CONFIG.TUSER_WIDTH {0} ] $m_axis_krnl

    set interfaces "$interfaces:s_axis_krnl:m_axis_krnl"

    # create segmenters
    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_krnl_in
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins sseg_krnl_in/ap_clk]
    connect_bd_net [get_bd_pins control/encore_aresetn] [get_bd_pins sseg_krnl_in/ap_rst_n]

    create_bd_cell -type ip -vlnv xilinx.com:hls:stream_segmenter:1.0 sseg_krnl_out
    connect_bd_net [get_bd_ports ap_clk] [get_bd_pins sseg_krnl_out/ap_clk]
    connect_bd_net [get_bd_pins control/encore_aresetn] [get_bd_pins sseg_krnl_out/ap_rst_n]

    connect_bd_intf_net [get_bd_intf_pins s_axis_krnl] [get_bd_intf_pins sseg_krnl_in/in_r]
    connect_bd_intf_net [get_bd_intf_pins sseg_krnl_in/out_r] [get_bd_intf_pins axis_switch_0/S04_AXIS]

    connect_bd_intf_net [get_bd_intf_pins sseg_krnl_in/cmd] [get_bd_intf_pins control/krnl_in_seg_cmd]
    connect_bd_intf_net [get_bd_intf_pins sseg_krnl_out/cmd] [get_bd_intf_pins control/krnl_out_seg_cmd]

    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M05_AXIS] [get_bd_intf_pins sseg_krnl_out/in_r]
    connect_bd_intf_net [get_bd_intf_pins control/krnl_out_seg_sts] [get_bd_intf_pins sseg_krnl_out/sts]

    # create bypass switch, for any ETH ingress data targeting the external kernels directly
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1
    set_property -dict [list  CONFIG.HAS_TLAST.VALUE_SRC USER \
                              CONFIG.HAS_TLAST {1} \
                              CONFIG.ARB_ON_TLAST {1}\
                              CONFIG.ARB_ALGORITHM {3} \
                              CONFIG.ARB_ON_MAX_XFERS {0} \
                              CONFIG.HAS_TSTRB.VALUE_SRC USER \
                              CONFIG.HAS_TSTRB {0} \
                              CONFIG.M00_AXIS_HIGHTDEST {0x000000ff} \
    ] [get_bd_cells axis_switch_1]

    # create subset converter to adjust TDEST on ingress path to kernels
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 krnl_ssc
    set_property -dict [list CONFIG.S_TDEST_WIDTH.VALUE_SRC USER CONFIG.M_TDEST_WIDTH.VALUE_SRC USER] [get_bd_cells krnl_ssc]
    set_property -dict [list CONFIG.S_TDEST_WIDTH {8} CONFIG.M_TDEST_WIDTH {8} CONFIG.TDEST_REMAP {tdest[7:0]-9}] [get_bd_cells krnl_ssc]

    connect_bd_intf_net [get_bd_intf_pins sseg_krnl_out/out_r] [get_bd_intf_pins axis_switch_1/S00_AXIS]
    connect_bd_intf_net [get_bd_intf_pins axis_switch_0/M09_AXIS] [get_bd_intf_pins krnl_ssc/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins krnl_ssc/M_AXIS] [get_bd_intf_pins axis_switch_1/S01_AXIS]
    connect_bd_intf_net [get_bd_intf_pins m_axis_krnl] [get_bd_intf_pins axis_switch_1/M00_AXIS]
    connect_bd_net [get_bd_pins ap_clk] [get_bd_pins axis_switch_1/aclk]
    connect_bd_net [get_bd_pins control/encore_aresetn] [get_bd_pins axis_switch_1/aresetn]
    connect_bd_net [get_bd_pins ap_clk] [get_bd_pins krnl_ssc/aclk]
    connect_bd_net [get_bd_pins control/encore_aresetn] [get_bd_pins krnl_ssc/aresetn]
  }

  # Create control interface connections
  connect_bd_intf_net -intf_net host_control [get_bd_intf_ports s_axi_control] [get_bd_intf_pins control/host_control]
  connect_bd_intf_net -intf_net encore_control [get_bd_intf_pins control_xbar/S00_AXI] [get_bd_intf_pins control/encore_control]
  connect_bd_intf_net [get_bd_intf_ports s_axis_call_req] [get_bd_intf_pins control/call_req]
  connect_bd_intf_net [get_bd_intf_pins control/call_ack] [get_bd_intf_ports m_axis_call_ack] 

  if { $debugLevel != 0 } {
    set bscan_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:bscan_rtl:1.0 bscan_0 ]
    connect_bd_intf_net -intf_net bscan_0 [get_bd_intf_ports bscan_0] [get_bd_intf_pins control/bscan_0]
  }

  # Create reset and clock connections
  connect_bd_net [get_bd_ports ap_clk] \
                 [get_bd_pins axis_switch_0/aclk] \
                 [get_bd_pins axis_switch_0/s_axi_ctrl_aclk] \
                 [get_bd_pins control/ap_clk] \
                 [get_bd_pins axi_gpio_tdest/s_axi_aclk] \
                 [get_bd_pins control_xbar/ACLK] \
                 [get_bd_pins control_xbar/S00_ACLK] \
                 [get_bd_pins control_xbar/M00_ACLK] \
                 [get_bd_pins control_xbar/M01_ACLK]
  connect_bd_net -net ap_rst_n [get_bd_ports ap_rst_n] [get_bd_pins control/ap_rst_n]
  connect_bd_net [get_bd_pins control/encore_aresetn] \
                 [get_bd_pins axis_switch_0/aresetn] \
                 [get_bd_pins axis_switch_0/s_axi_ctrl_aresetn] \
                 [get_bd_pins axi_gpio_tdest/s_axi_aresetn] \
                 [get_bd_pins control_xbar/ARESETN] \
                 [get_bd_pins control_xbar/S00_ARESETN] \
                 [get_bd_pins control_xbar/M00_ARESETN] \
                 [get_bd_pins control_xbar/M01_ARESETN]

  set_property -dict [ list CONFIG.ASSOCIATED_BUSIF $interfaces ] $ap_clk

  save_bd_design

  # put everything into proper hierarchy
  create_bd_cell -type hier routing_subsystem
  move_bd_cells [get_bd_cells routing_subsystem] [get_bd_cells sseg*] 
  move_bd_cells [get_bd_cells routing_subsystem] [get_bd_cells axis_switch*]
  move_bd_cells [get_bd_cells routing_subsystem] [get_bd_cells krnl_ssc]

  create_bd_cell -type hier cclo
  move_bd_cells [get_bd_cells cclo] [get_bd_cells control] 
  move_bd_cells [get_bd_cells cclo] [get_bd_cells routing_subsystem] 
  move_bd_cells [get_bd_cells cclo] [get_bd_cells control_xbar]
  catch { move_bd_cells [get_bd_cells cclo] [get_bd_cells dma_0] }
  catch { move_bd_cells [get_bd_cells cclo] [get_bd_cells dma_1] }
  catch { move_bd_cells [get_bd_cells cclo] [get_bd_cells eth_tx_subsystem] }
  catch { move_bd_cells [get_bd_cells cclo] [get_bd_cells eth_rx_subsystem] }
  catch { move_bd_cells [get_bd_cells cclo] [get_bd_cells *_dmawr_*] }

  validate_bd_design
  save_bd_design
}

