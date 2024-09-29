
# /*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
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

set nettype [lindex $::argv 0]
set build_dir [lindex $::argv 1]
open_project "$build_dir/test_config_0/user_c0_0/test.xpr"
update_compile_order -fileset sources_1
create_bd_design "accl_bd"
update_compile_order -fileset sources_1
update_ip_catalog



  # Create interface ports
  set S00_AXI_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI_0 ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH {14} \
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
   ] $S00_AXI_0

  set cyt_cq_rd_sts_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_cq_rd_sts_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {4} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_cq_rd_sts_0

  set cyt_cq_wr_sts_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_cq_wr_sts_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {4} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_cq_wr_sts_0

  set cyt_rq_rd [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rq_rd ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {16} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rq_rd

  set cyt_rq_wr [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rq_wr ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {0} \
   CONFIG.HAS_TLAST {0} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {16} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rq_wr

  set cyt_rreq_recv_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rreq_recv_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {0} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rreq_recv_0

  set cyt_rreq_recv_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rreq_recv_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rreq_recv_1

  set cyt_rreq_send_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rreq_send_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_rreq_send_0

  set cyt_rreq_send_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rreq_send_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_rreq_send_1

  set cyt_rrsp_recv_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rrsp_recv_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rrsp_recv_0

  set cyt_rrsp_recv_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rrsp_recv_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $cyt_rrsp_recv_1

  set cyt_rrsp_send_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rrsp_send_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_rrsp_send_0

  set cyt_rrsp_send_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_rrsp_send_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_rrsp_send_1

  set cyt_sq_rd_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_sq_rd_cmd ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_sq_rd_cmd

  set cyt_sq_wr_cmd [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cyt_sq_wr_cmd ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $cyt_sq_wr_cmd

  set m_axis_card_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_card_0

  set m_axis_card_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_card_1

  set m_axis_card_2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_card_2 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_card_2

  set m_axis_host_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_host_0

  set m_axis_host_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_host_1

  set m_axis_host_2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_host_2 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   ] $m_axis_host_2

  set s_axis_card_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_card_0

  set s_axis_card_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_card_1

  set s_axis_card_2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_card_2 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_card_2

  set s_axis_host_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_host_0

  set s_axis_host_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_host_1

  set s_axis_host_2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_host_2 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {250000000} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.HAS_TREADY {1} \
   CONFIG.HAS_TSTRB {0} \
   CONFIG.LAYERED_METADATA {undef} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {0} \
   CONFIG.TID_WIDTH {0} \
   CONFIG.TUSER_WIDTH {0} \
   ] $s_axis_host_2


  # Create ports
  set ap_clk_0 [ create_bd_port -dir I -type clk -freq_hz 250000000 ap_clk_0 ]
  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {cyt_cq_wr_sts_0:cyt_cq_rd_sts_0:cyt_sq_wr_cmd:cyt_sq_rd_cmd:m_axis_host_2:m_axis_card_2:s_axis_host_2:s_axis_card_2:cyt_rq_wr:m_axis_host_0:m_axis_host_1:m_axis_card_0:m_axis_card_1:s_axis_host_0:s_axis_host_1:s_axis_card_0:s_axis_card_1:S00_AXI_0:cyt_rreq_send_0:cyt_rreq_send_1:cyt_rrsp_recv_0:cyt_rrsp_recv_1:cyt_rrsp_send_0:cyt_rrsp_send_1:cyt_rq_rd:cyt_rreq_recv_0:cyt_rreq_recv_1} \
 ] $ap_clk_0
  set ap_rst_n_0 [ create_bd_port -dir I -type rst ap_rst_n_0 ]

  # Create instance: axis_data_fifo_0, and set properties
  set axis_data_fifo_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_0 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_0

  # Create instance: axis_data_fifo_1, and set properties
  set axis_data_fifo_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_1 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_1

  # Create instance: axis_data_fifo_2, and set properties
  set axis_data_fifo_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_2 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_2

  # Create instance: axis_data_fifo_3, and set properties
  set axis_data_fifo_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_3 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_3

  # Create instance: axis_data_fifo_4, and set properties
  set axis_data_fifo_4 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_4 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_4

  # Create instance: axis_data_fifo_5, and set properties
  set axis_data_fifo_5 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_5 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_5

  # Create instance: axis_data_fifo_6, and set properties
  set axis_data_fifo_6 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_6 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_6

  # Create instance: axis_data_fifo_7, and set properties
  set axis_data_fifo_7 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_7 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_7

  # Create instance: axis_data_fifo_8, and set properties
  set axis_data_fifo_8 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_8 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_8

  # Create instance: axis_data_fifo_9, and set properties
  set axis_data_fifo_9 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_9 ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {16} \
 ] $axis_data_fifo_9

  # Create instance: axis_register_slice_0, and set properties
  set axis_register_slice_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_0 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_0

  # Create instance: axis_register_slice_1, and set properties
  set axis_register_slice_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_1 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_1

  # Create instance: axis_register_slice_2, and set properties
  set axis_register_slice_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_2 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_2

  # Create instance: axis_register_slice_3, and set properties
  set axis_register_slice_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_3 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_3

  # Create instance: axis_register_slice_4, and set properties
  set axis_register_slice_4 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_4 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_4

  # Create instance: axis_register_slice_5, and set properties
  set axis_register_slice_5 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_5 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_5

  # Create instance: axis_register_slice_6, and set properties
  set axis_register_slice_6 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_6 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_6

  # Create instance: axis_register_slice_7, and set properties
  set axis_register_slice_7 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_7 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_7

  # Create instance: axis_register_slice_8, and set properties
  set axis_register_slice_8 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_8 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_8

  # Create instance: axis_register_slice_9, and set properties
  set axis_register_slice_9 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_9 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_9

  # Create instance: axis_register_slice_10, and set properties
  set axis_register_slice_10 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_10 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_10

  # Create instance: axis_register_slice_11, and set properties
  set axis_register_slice_11 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_11 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_11

  # Create instance: axis_register_slice_12, and set properties
  set axis_register_slice_12 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_12 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_12

  # Create instance: axis_register_slice_13, and set properties
  set axis_register_slice_13 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_13 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_13

  # Create instance: axis_register_slice_14, and set properties
  set axis_register_slice_14 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 axis_register_slice_14 ]
  set_property -dict [ list \
   CONFIG.NUM_SLR_CROSSINGS {0} \
   CONFIG.REG_CONFIG {16} \
 ] $axis_register_slice_14

  # Create instance: axis_switch_1_to_2_inst_0, and set properties
  set axis_switch_1_to_2_inst_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_0 ]
  set_property -dict [ list \
   CONFIG.DECODER_REG {1} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {1} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {8} \
 ] $axis_switch_1_to_2_inst_0

  # Create instance: axis_switch_1_to_2_inst_1, and set properties
  set axis_switch_1_to_2_inst_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_1 ]
  set_property -dict [ list \
   CONFIG.DECODER_REG {1} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {1} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {8} \
 ] $axis_switch_1_to_2_inst_1

  # Create instance: axis_switch_1_to_2_inst_2, and set properties
  set axis_switch_1_to_2_inst_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_2 ]
  set_property -dict [ list \
   CONFIG.DECODER_REG {1} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {1} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {8} \
 ] $axis_switch_1_to_2_inst_2

  # Create instance: axis_switch_1_to_2_inst_3, and set properties
  set axis_switch_1_to_2_inst_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_1_to_2_inst_3 ]
  set_property -dict [ list \
   CONFIG.DECODER_REG {1} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {1} \
   CONFIG.TDATA_NUM_BYTES {64} \
   CONFIG.TDEST_WIDTH {8} \
 ] $axis_switch_1_to_2_inst_3

  # Create instance: axis_switch_2_to_1_inst_0, and set properties
  set axis_switch_2_to_1_inst_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_0 ]
  set_property -dict [ list \
   CONFIG.ARB_ON_MAX_XFERS {0} \
   CONFIG.ARB_ON_TLAST {1} \
   CONFIG.DECODER_REG {0} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {2} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $axis_switch_2_to_1_inst_0

  # Create instance: axis_switch_2_to_1_inst_1, and set properties
  set axis_switch_2_to_1_inst_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_1 ]
  set_property -dict [ list \
   CONFIG.ARB_ON_MAX_XFERS {0} \
   CONFIG.ARB_ON_TLAST {1} \
   CONFIG.DECODER_REG {0} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {2} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $axis_switch_2_to_1_inst_1

  # Create instance: axis_switch_2_to_1_inst_2, and set properties
  set axis_switch_2_to_1_inst_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_switch:1.1 axis_switch_2_to_1_inst_2 ]
  set_property -dict [ list \
   CONFIG.ARB_ON_MAX_XFERS {0} \
   CONFIG.ARB_ON_TLAST {1} \
   CONFIG.DECODER_REG {0} \
   CONFIG.HAS_TKEEP {1} \
   CONFIG.HAS_TLAST {1} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {2} \
   CONFIG.TDATA_NUM_BYTES {64} \
 ] $axis_switch_2_to_1_inst_2

  # Create instance: ccl_offload_0, and set properties
  set ccl_offload_0 [ create_bd_cell -type ip -vlnv Xilinx:ACCL:ccl_offload:1.0 ccl_offload_0 ]

  # Create instance: cclo_sq_adapter_0, and set properties
  set cclo_sq_adapter_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:cclo_sq_adapter:1.0 cclo_sq_adapter_0 ]

  # Create instance: cyt_cq_dm_sts_conver_0, and set properties
  set cyt_cq_dm_sts_conver_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_cq_dm_sts_converter:1.0 cyt_cq_dm_sts_conver_0 ]

  # Create instance: cyt_cq_dm_sts_conver_1, and set properties
  set cyt_cq_dm_sts_conver_1 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_cq_dm_sts_converter:1.0 cyt_cq_dm_sts_conver_1 ]

  # Create instance: cyt_dma_sq_adapter_0, and set properties
  set cyt_dma_sq_adapter_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_dma_sq_adapter:1.0 cyt_dma_sq_adapter_0 ]

  # Create instance: cyt_rdma_arbiter_0, and set properties
  set cyt_rdma_arbiter_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:cyt_rdma_arbiter:1.0 cyt_rdma_arbiter_0 ]

  # Create instance: hostctrl_0, and set properties
  set hostctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:hostctrl:1.0 hostctrl_0 ]

  # Create instance: reduce_ops_0, and set properties
  set reduce_ops_0 [ create_bd_cell -type ip -vlnv xilinx.com:ACCL:reduce_ops:1.0 reduce_ops_0 ]

  # Create instance: rst_ap_clk_0_250M, and set properties
  set rst_ap_clk_0_250M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ap_clk_0_250M ]

  # Create instance: smartconnect_0, and set properties
  set smartconnect_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0 ]
  set_property -dict [ list \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_0

  # Create instance: system_ila_0, and set properties
  set system_ila_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_0 ]
  set_property -dict [ list \
   CONFIG.C_INPUT_PIPE_STAGES {2} \
   CONFIG.C_MON_TYPE {INTERFACE} \
   CONFIG.C_NUM_MONITOR_SLOTS {16} \
   CONFIG.C_SLOT_0_APC_EN {0} \
   CONFIG.C_SLOT_0_AXI_AR_SEL_DATA {1} \
   CONFIG.C_SLOT_0_AXI_AR_SEL_TRIG {1} \
   CONFIG.C_SLOT_0_AXI_AW_SEL_DATA {1} \
   CONFIG.C_SLOT_0_AXI_AW_SEL_TRIG {1} \
   CONFIG.C_SLOT_0_AXI_B_SEL_DATA {1} \
   CONFIG.C_SLOT_0_AXI_B_SEL_TRIG {1} \
   CONFIG.C_SLOT_0_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_0_AXI_R_SEL_DATA {1} \
   CONFIG.C_SLOT_0_AXI_R_SEL_TRIG {1} \
   CONFIG.C_SLOT_0_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_0_AXI_W_SEL_DATA {1} \
   CONFIG.C_SLOT_0_AXI_W_SEL_TRIG {1} \
   CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_0_TYPE {0} \
   CONFIG.C_SLOT_10_APC_EN {0} \
   CONFIG.C_SLOT_10_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_10_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_10_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_10_TYPE {0} \
   CONFIG.C_SLOT_11_APC_EN {0} \
   CONFIG.C_SLOT_11_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_11_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_11_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_11_TYPE {0} \
   CONFIG.C_SLOT_12_APC_EN {0} \
   CONFIG.C_SLOT_12_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_12_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_12_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_12_TYPE {0} \
   CONFIG.C_SLOT_13_APC_EN {0} \
   CONFIG.C_SLOT_13_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_13_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_13_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_13_TYPE {0} \
   CONFIG.C_SLOT_14_APC_EN {0} \
   CONFIG.C_SLOT_14_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_14_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_14_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_14_TYPE {0} \
   CONFIG.C_SLOT_15_APC_EN {0} \
   CONFIG.C_SLOT_15_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_15_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_15_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_1_APC_EN {0} \
   CONFIG.C_SLOT_1_AXI_AR_SEL_DATA {1} \
   CONFIG.C_SLOT_1_AXI_AR_SEL_TRIG {1} \
   CONFIG.C_SLOT_1_AXI_AW_SEL_DATA {1} \
   CONFIG.C_SLOT_1_AXI_AW_SEL_TRIG {1} \
   CONFIG.C_SLOT_1_AXI_B_SEL_DATA {1} \
   CONFIG.C_SLOT_1_AXI_B_SEL_TRIG {1} \
   CONFIG.C_SLOT_1_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_1_AXI_R_SEL_DATA {1} \
   CONFIG.C_SLOT_1_AXI_R_SEL_TRIG {1} \
   CONFIG.C_SLOT_1_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_1_AXI_W_SEL_DATA {1} \
   CONFIG.C_SLOT_1_AXI_W_SEL_TRIG {1} \
   CONFIG.C_SLOT_1_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_1_TYPE {0} \
   CONFIG.C_SLOT_2_APC_EN {0} \
   CONFIG.C_SLOT_2_AXI_AR_SEL_DATA {1} \
   CONFIG.C_SLOT_2_AXI_AR_SEL_TRIG {1} \
   CONFIG.C_SLOT_2_AXI_AW_SEL_DATA {1} \
   CONFIG.C_SLOT_2_AXI_AW_SEL_TRIG {1} \
   CONFIG.C_SLOT_2_AXI_B_SEL_DATA {1} \
   CONFIG.C_SLOT_2_AXI_B_SEL_TRIG {1} \
   CONFIG.C_SLOT_2_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_2_AXI_R_SEL_DATA {1} \
   CONFIG.C_SLOT_2_AXI_R_SEL_TRIG {1} \
   CONFIG.C_SLOT_2_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_2_AXI_W_SEL_DATA {1} \
   CONFIG.C_SLOT_2_AXI_W_SEL_TRIG {1} \
   CONFIG.C_SLOT_2_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_2_TYPE {0} \
   CONFIG.C_SLOT_3_APC_EN {0} \
   CONFIG.C_SLOT_3_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_3_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_3_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_3_TYPE {0} \
   CONFIG.C_SLOT_4_APC_EN {0} \
   CONFIG.C_SLOT_4_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_4_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_4_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_4_TYPE {0} \
   CONFIG.C_SLOT_5_APC_EN {0} \
   CONFIG.C_SLOT_5_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_5_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_5_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_5_TYPE {0} \
   CONFIG.C_SLOT_6_APC_EN {0} \
   CONFIG.C_SLOT_6_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_6_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_6_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_6_TYPE {0} \
   CONFIG.C_SLOT_7_APC_EN {0} \
   CONFIG.C_SLOT_7_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_7_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_7_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_7_TYPE {0} \
   CONFIG.C_SLOT_8_APC_EN {0} \
   CONFIG.C_SLOT_8_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_8_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_8_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_8_TYPE {0} \
   CONFIG.C_SLOT_9_APC_EN {0} \
   CONFIG.C_SLOT_9_AXI_DATA_SEL {1} \
   CONFIG.C_SLOT_9_AXI_TRIG_SEL {1} \
   CONFIG.C_SLOT_9_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
   CONFIG.C_SLOT_9_TYPE {0} \
 ] $system_ila_0

  # Create instance: xlconstant_0, and set properties
  set xlconstant_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {2} \
 ] $xlconstant_0

  # Create instance: xlconstant_1, and set properties
  set xlconstant_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_1 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {2} \
 ] $xlconstant_1

  # Create instance: xlconstant_2, and set properties
  set xlconstant_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_2 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {2} \
 ] $xlconstant_2

  # Create interface connections
  connect_bd_intf_net -intf_net S00_AXI_0_1 [get_bd_intf_ports S00_AXI_0] [get_bd_intf_pins smartconnect_0/S00_AXI]
  connect_bd_intf_net -intf_net axis_data_fifo_0_M_AXIS [get_bd_intf_ports cyt_rrsp_send_0] [get_bd_intf_pins axis_data_fifo_0/M_AXIS]
  connect_bd_intf_net -intf_net axis_data_fifo_1_M_AXIS [get_bd_intf_ports cyt_rrsp_send_1] [get_bd_intf_pins axis_data_fifo_1/M_AXIS]
  connect_bd_intf_net -intf_net axis_data_fifo_2_M_AXIS [get_bd_intf_pins axis_data_fifo_2/M_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cyt_rq_wr_cmd]
  connect_bd_intf_net -intf_net axis_data_fifo_3_M_AXIS [get_bd_intf_pins axis_data_fifo_3/M_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cyt_rq_rd_cmd]
  connect_bd_intf_net -intf_net axis_data_fifo_4_M_AXIS [get_bd_intf_pins axis_data_fifo_4/M_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cclo_sq_wr_cmd]
  connect_bd_intf_net -intf_net axis_data_fifo_5_M_AXIS [get_bd_intf_pins axis_data_fifo_5/M_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cclo_sq_rd_cmd]
  connect_bd_intf_net -intf_net axis_data_fifo_6_M_AXIS [get_bd_intf_pins axis_data_fifo_6/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm1_meta]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_data_fifo_6_M_AXIS] [get_bd_intf_pins axis_data_fifo_6/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_0_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_data_fifo_6_M_AXIS]
  connect_bd_intf_net -intf_net axis_data_fifo_7_M_AXIS [get_bd_intf_pins axis_data_fifo_7/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm0_meta]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_data_fifo_7_M_AXIS] [get_bd_intf_pins axis_data_fifo_7/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_1_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_data_fifo_7_M_AXIS]
  connect_bd_intf_net -intf_net axis_data_fifo_8_M_AXIS [get_bd_intf_pins axis_data_fifo_8/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm1_meta]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_data_fifo_8_M_AXIS] [get_bd_intf_pins axis_data_fifo_8/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_6_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_data_fifo_8_M_AXIS]
  connect_bd_intf_net -intf_net axis_data_fifo_9_M_AXIS [get_bd_intf_pins axis_data_fifo_9/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm0_meta]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_data_fifo_9_M_AXIS] [get_bd_intf_pins axis_data_fifo_9/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_7_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_data_fifo_9_M_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_0_M_AXIS [get_bd_intf_pins axis_register_slice_0/M_AXIS] [get_bd_intf_pins cyt_rdma_arbiter_0/s_axis_0]
  connect_bd_intf_net -intf_net axis_register_slice_10_M_AXIS [get_bd_intf_ports cyt_sq_rd_cmd] [get_bd_intf_pins axis_register_slice_10/M_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_11_M_AXIS [get_bd_intf_pins axis_register_slice_11/M_AXIS] [get_bd_intf_pins axis_switch_1_to_2_inst_3/S00_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_12_M_AXIS [get_bd_intf_pins axis_register_slice_12/M_AXIS] [get_bd_intf_pins axis_switch_1_to_2_inst_0/S00_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_13_M_AXIS [get_bd_intf_pins axis_register_slice_13/M_AXIS] [get_bd_intf_pins axis_switch_1_to_2_inst_1/S00_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_14_M_AXIS [get_bd_intf_pins axis_register_slice_14/M_AXIS] [get_bd_intf_pins axis_switch_1_to_2_inst_2/S00_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_1_M_AXIS [get_bd_intf_pins axis_register_slice_1/M_AXIS] [get_bd_intf_pins cyt_rdma_arbiter_0/s_axis_1]
  connect_bd_intf_net -intf_net axis_register_slice_2_M_AXIS [get_bd_intf_pins axis_register_slice_2/M_AXIS] [get_bd_intf_pins cyt_rdma_arbiter_0/s_meta]
  connect_bd_intf_net -intf_net axis_register_slice_3_M_AXIS [get_bd_intf_pins axis_register_slice_3/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/cq_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_register_slice_3_M_AXIS] [get_bd_intf_pins axis_register_slice_3/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_8_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_register_slice_3_M_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_4_M_AXIS [get_bd_intf_pins axis_register_slice_4/M_AXIS] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/cq_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets axis_register_slice_4_M_AXIS] [get_bd_intf_pins axis_register_slice_4/M_AXIS] [get_bd_intf_pins system_ila_0/SLOT_9_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets axis_register_slice_4_M_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_5_M_AXIS [get_bd_intf_pins axis_data_fifo_3/S_AXIS] [get_bd_intf_pins axis_register_slice_5/M_AXIS]
  connect_bd_intf_net -intf_net axis_register_slice_6_M_AXIS [get_bd_intf_pins axis_register_slice_6/M_AXIS] [get_bd_intf_pins cclo_sq_adapter_0/s_axis_cyt]
  connect_bd_intf_net -intf_net axis_register_slice_7_M_AXIS [get_bd_intf_pins axis_register_slice_7/M_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s]
  connect_bd_intf_net -intf_net axis_register_slice_8_M_AXIS [get_bd_intf_pins axis_register_slice_8/M_AXIS] [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s]
  connect_bd_intf_net -intf_net axis_register_slice_9_M_AXIS [get_bd_intf_ports cyt_sq_wr_cmd] [get_bd_intf_pins axis_register_slice_9/M_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_0_M00_AXIS [get_bd_intf_ports m_axis_card_0] [get_bd_intf_pins axis_switch_1_to_2_inst_0/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_0_M01_AXIS [get_bd_intf_ports m_axis_host_0] [get_bd_intf_pins axis_switch_1_to_2_inst_0/M01_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_1_M00_AXIS [get_bd_intf_ports m_axis_card_1] [get_bd_intf_pins axis_switch_1_to_2_inst_1/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_1_M01_AXIS [get_bd_intf_ports m_axis_host_1] [get_bd_intf_pins axis_switch_1_to_2_inst_1/M01_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_2_M00_AXIS [get_bd_intf_ports m_axis_card_2] [get_bd_intf_pins axis_switch_1_to_2_inst_2/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_2_M01_AXIS [get_bd_intf_ports m_axis_host_2] [get_bd_intf_pins axis_switch_1_to_2_inst_2/M01_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_3_M00_AXIS [get_bd_intf_ports cyt_rreq_send_0] [get_bd_intf_pins axis_switch_1_to_2_inst_3/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_1_to_2_inst_3_M01_AXIS [get_bd_intf_ports cyt_rreq_send_1] [get_bd_intf_pins axis_switch_1_to_2_inst_3/M01_AXIS]
  connect_bd_intf_net -intf_net axis_switch_2_to_1_inst_0_M00_AXIS [get_bd_intf_pins axis_register_slice_7/S_AXIS] [get_bd_intf_pins axis_switch_2_to_1_inst_0/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_2_to_1_inst_1_M00_AXIS [get_bd_intf_pins axis_register_slice_8/S_AXIS] [get_bd_intf_pins axis_switch_2_to_1_inst_1/M00_AXIS]
  connect_bd_intf_net -intf_net axis_switch_2_to_1_inst_2_M00_AXIS [get_bd_intf_pins axis_register_slice_6/S_AXIS] [get_bd_intf_pins axis_switch_2_to_1_inst_2/M00_AXIS]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_arith_op0 [get_bd_intf_pins ccl_offload_0/m_axis_arith_op0] [get_bd_intf_pins reduce_ops_0/in0]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_arith_op1 [get_bd_intf_pins ccl_offload_0/m_axis_arith_op1] [get_bd_intf_pins reduce_ops_0/in1]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_call_ack [get_bd_intf_pins ccl_offload_0/m_axis_call_ack] [get_bd_intf_pins hostctrl_0/sts]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_compression0 [get_bd_intf_pins ccl_offload_0/m_axis_compression0] [get_bd_intf_pins ccl_offload_0/s_axis_compression0]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_compression1 [get_bd_intf_pins ccl_offload_0/m_axis_compression1] [get_bd_intf_pins ccl_offload_0/s_axis_compression1]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_compression2 [get_bd_intf_pins ccl_offload_0/m_axis_compression2] [get_bd_intf_pins ccl_offload_0/s_axis_compression2]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma0_mm2s_cmd [get_bd_intf_pins ccl_offload_0/m_axis_dma0_mm2s_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_mm2s_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets ccl_offload_0_m_axis_dma0_mm2s_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_mm2s_cmd] [get_bd_intf_pins system_ila_0/SLOT_2_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets ccl_offload_0_m_axis_dma0_mm2s_cmd]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma0_s2mm [get_bd_intf_pins axis_register_slice_12/S_AXIS] [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma0_s2mm_cmd [get_bd_intf_pins ccl_offload_0/m_axis_dma0_s2mm_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_s2mm_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets ccl_offload_0_m_axis_dma0_s2mm_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_s2mm_cmd] [get_bd_intf_pins system_ila_0/SLOT_3_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets ccl_offload_0_m_axis_dma0_s2mm_cmd]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma1_mm2s_cmd [get_bd_intf_pins ccl_offload_0/m_axis_dma1_mm2s_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_mm2s_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets ccl_offload_0_m_axis_dma1_mm2s_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_mm2s_cmd] [get_bd_intf_pins system_ila_0/SLOT_4_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets ccl_offload_0_m_axis_dma1_mm2s_cmd]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma1_s2mm [get_bd_intf_pins axis_register_slice_13/S_AXIS] [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_dma1_s2mm_cmd [get_bd_intf_pins ccl_offload_0/m_axis_dma1_s2mm_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_s2mm_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets ccl_offload_0_m_axis_dma1_s2mm_cmd] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_s2mm_cmd] [get_bd_intf_pins system_ila_0/SLOT_5_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets ccl_offload_0_m_axis_dma1_s2mm_cmd]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_eth_tx_data [get_bd_intf_pins ccl_offload_0/m_axis_eth_tx_data] [get_bd_intf_pins cclo_sq_adapter_0/s_axis_cclo]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_krnl [get_bd_intf_pins ccl_offload_0/m_axis_krnl] [get_bd_intf_pins ccl_offload_0/s_axis_krnl]
  connect_bd_intf_net -intf_net ccl_offload_0_m_axis_rdma_sq [get_bd_intf_pins ccl_offload_0/m_axis_rdma_sq] [get_bd_intf_pins cclo_sq_adapter_0/cclo_sq]
  connect_bd_intf_net -intf_net cclo_sq_adapter_0_cyt_sq_rd [get_bd_intf_pins axis_data_fifo_5/S_AXIS] [get_bd_intf_pins cclo_sq_adapter_0/cyt_sq_rd]
  connect_bd_intf_net -intf_net cclo_sq_adapter_0_cyt_sq_wr [get_bd_intf_pins axis_data_fifo_4/S_AXIS] [get_bd_intf_pins cclo_sq_adapter_0/cyt_sq_wr]
  connect_bd_intf_net -intf_net cclo_sq_adapter_0_m_axis_cyt [get_bd_intf_pins axis_register_slice_11/S_AXIS] [get_bd_intf_pins cclo_sq_adapter_0/m_axis_cyt]
  connect_bd_intf_net -intf_net cyt_cq_dm_sts_conver_0_dm0_sts [get_bd_intf_pins ccl_offload_0/s_axis_dma0_s2mm_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm0_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_cq_dm_sts_conver_0_dm0_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm0_sts] [get_bd_intf_pins system_ila_0/SLOT_12_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_cq_dm_sts_conver_0_dm0_sts]
  connect_bd_intf_net -intf_net cyt_cq_dm_sts_conver_0_dm1_sts [get_bd_intf_pins ccl_offload_0/s_axis_dma1_s2mm_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm1_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_cq_dm_sts_conver_0_dm1_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_0/dm1_sts] [get_bd_intf_pins system_ila_0/SLOT_13_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_cq_dm_sts_conver_0_dm1_sts]
  connect_bd_intf_net -intf_net cyt_cq_dm_sts_conver_1_dm0_sts [get_bd_intf_pins ccl_offload_0/s_axis_dma0_mm2s_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm0_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_cq_dm_sts_conver_1_dm0_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm0_sts] [get_bd_intf_pins system_ila_0/SLOT_14_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_cq_dm_sts_conver_1_dm0_sts]
  connect_bd_intf_net -intf_net cyt_cq_dm_sts_conver_1_dm1_sts [get_bd_intf_pins ccl_offload_0/s_axis_dma1_mm2s_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm1_sts]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_cq_dm_sts_conver_1_dm1_sts] [get_bd_intf_pins cyt_cq_dm_sts_conver_1/dm1_sts] [get_bd_intf_pins system_ila_0/SLOT_15_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_cq_dm_sts_conver_1_dm1_sts]
  connect_bd_intf_net -intf_net cyt_cq_rd_sts_0_1 [get_bd_intf_ports cyt_cq_rd_sts_0] [get_bd_intf_pins axis_register_slice_4/S_AXIS]
  connect_bd_intf_net -intf_net cyt_cq_wr_sts_0_1 [get_bd_intf_ports cyt_cq_wr_sts_0] [get_bd_intf_pins axis_register_slice_3/S_AXIS]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_cyt_sq_rd_cmd [get_bd_intf_pins axis_register_slice_10/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cyt_sq_rd_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_dma_sq_adapter_0_cyt_sq_rd_cmd] [get_bd_intf_pins axis_register_slice_10/S_AXIS] [get_bd_intf_pins system_ila_0/SLOT_10_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_dma_sq_adapter_0_cyt_sq_rd_cmd]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_cyt_sq_wr_cmd [get_bd_intf_pins axis_register_slice_9/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/cyt_sq_wr_cmd]
connect_bd_intf_net -intf_net [get_bd_intf_nets cyt_dma_sq_adapter_0_cyt_sq_wr_cmd] [get_bd_intf_pins axis_register_slice_9/S_AXIS] [get_bd_intf_pins system_ila_0/SLOT_11_AXIS]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets cyt_dma_sq_adapter_0_cyt_sq_wr_cmd]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_dma0_mm2s_meta [get_bd_intf_pins axis_data_fifo_7/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_mm2s_meta]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_dma0_s2mm_meta [get_bd_intf_pins axis_data_fifo_9/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma0_s2mm_meta]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_dma1_mm2s_meta [get_bd_intf_pins axis_data_fifo_6/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_mm2s_meta]
  connect_bd_intf_net -intf_net cyt_dma_sq_adapter_0_dma1_s2mm_meta [get_bd_intf_pins axis_data_fifo_8/S_AXIS] [get_bd_intf_pins cyt_dma_sq_adapter_0/dma1_s2mm_meta]
  connect_bd_intf_net -intf_net cyt_rdma_arbiter_0_m_axis_0 [get_bd_intf_pins ccl_offload_0/s_axis_eth_rx_data] [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_0]
  connect_bd_intf_net -intf_net cyt_rdma_arbiter_0_m_axis_1 [get_bd_intf_pins axis_register_slice_14/S_AXIS] [get_bd_intf_pins cyt_rdma_arbiter_0/m_axis_1]
  connect_bd_intf_net -intf_net cyt_rdma_arbiter_0_m_meta_0 [get_bd_intf_pins ccl_offload_0/s_axis_eth_notification] [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_0]
  connect_bd_intf_net -intf_net cyt_rdma_arbiter_0_m_meta_1 [get_bd_intf_pins axis_data_fifo_2/S_AXIS] [get_bd_intf_pins cyt_rdma_arbiter_0/m_meta_1]
  connect_bd_intf_net -intf_net cyt_rq_rd_1 [get_bd_intf_ports cyt_rq_rd] [get_bd_intf_pins axis_register_slice_5/S_AXIS]
  connect_bd_intf_net -intf_net cyt_rq_wr_1 [get_bd_intf_ports cyt_rq_wr] [get_bd_intf_pins axis_register_slice_2/S_AXIS]
  connect_bd_intf_net -intf_net cyt_rreq_recv_0_1 [get_bd_intf_ports cyt_rreq_recv_0] [get_bd_intf_pins axis_switch_2_to_1_inst_2/S00_AXIS]
  connect_bd_intf_net -intf_net cyt_rreq_recv_1_1 [get_bd_intf_ports cyt_rreq_recv_1] [get_bd_intf_pins axis_switch_2_to_1_inst_2/S01_AXIS]
  connect_bd_intf_net -intf_net cyt_rrsp_recv_0_1 [get_bd_intf_ports cyt_rrsp_recv_0] [get_bd_intf_pins axis_register_slice_0/S_AXIS]
  connect_bd_intf_net -intf_net cyt_rrsp_recv_1_1 [get_bd_intf_ports cyt_rrsp_recv_1] [get_bd_intf_pins axis_register_slice_1/S_AXIS]
  connect_bd_intf_net -intf_net hostctrl_0_cmd [get_bd_intf_pins ccl_offload_0/s_axis_call_req] [get_bd_intf_pins hostctrl_0/cmd]
  connect_bd_intf_net -intf_net reduce_ops_0_out_r [get_bd_intf_pins ccl_offload_0/s_axis_arith_res] [get_bd_intf_pins reduce_ops_0/out_r]
  connect_bd_intf_net -intf_net s_axis_card_0_1 [get_bd_intf_ports s_axis_card_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S01_AXIS]
  connect_bd_intf_net -intf_net s_axis_card_1_1 [get_bd_intf_ports s_axis_card_1] [get_bd_intf_pins axis_switch_2_to_1_inst_1/S01_AXIS]
  connect_bd_intf_net -intf_net s_axis_card_2_1 [get_bd_intf_ports s_axis_card_2] [get_bd_intf_pins axis_data_fifo_0/S_AXIS]
  connect_bd_intf_net -intf_net s_axis_host_0_1 [get_bd_intf_ports s_axis_host_0] [get_bd_intf_pins axis_switch_2_to_1_inst_0/S00_AXIS]
  connect_bd_intf_net -intf_net s_axis_host_1_1 [get_bd_intf_ports s_axis_host_1] [get_bd_intf_pins axis_switch_2_to_1_inst_1/S00_AXIS]
  connect_bd_intf_net -intf_net s_axis_host_2_1 [get_bd_intf_ports s_axis_host_2] [get_bd_intf_pins axis_data_fifo_1/S_AXIS]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins hostctrl_0/s_axi_control] [get_bd_intf_pins smartconnect_0/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M01_AXI [get_bd_intf_pins ccl_offload_0/s_axi_control] [get_bd_intf_pins smartconnect_0/M01_AXI]

  # Create port connections
  connect_bd_net -net ap_clk_0_1 [get_bd_ports ap_clk_0] [get_bd_pins axis_data_fifo_0/s_axis_aclk] [get_bd_pins axis_data_fifo_1/s_axis_aclk] [get_bd_pins axis_data_fifo_2/s_axis_aclk] [get_bd_pins axis_data_fifo_3/s_axis_aclk] [get_bd_pins axis_data_fifo_4/s_axis_aclk] [get_bd_pins axis_data_fifo_5/s_axis_aclk] [get_bd_pins axis_data_fifo_6/s_axis_aclk] [get_bd_pins axis_data_fifo_7/s_axis_aclk] [get_bd_pins axis_data_fifo_8/s_axis_aclk] [get_bd_pins axis_data_fifo_9/s_axis_aclk] [get_bd_pins axis_register_slice_0/aclk] [get_bd_pins axis_register_slice_1/aclk] [get_bd_pins axis_register_slice_10/aclk] [get_bd_pins axis_register_slice_11/aclk] [get_bd_pins axis_register_slice_12/aclk] [get_bd_pins axis_register_slice_13/aclk] [get_bd_pins axis_register_slice_14/aclk] [get_bd_pins axis_register_slice_2/aclk] [get_bd_pins axis_register_slice_3/aclk] [get_bd_pins axis_register_slice_4/aclk] [get_bd_pins axis_register_slice_5/aclk] [get_bd_pins axis_register_slice_6/aclk] [get_bd_pins axis_register_slice_7/aclk] [get_bd_pins axis_register_slice_8/aclk] [get_bd_pins axis_register_slice_9/aclk] [get_bd_pins axis_switch_1_to_2_inst_0/aclk] [get_bd_pins axis_switch_1_to_2_inst_1/aclk] [get_bd_pins axis_switch_1_to_2_inst_2/aclk] [get_bd_pins axis_switch_1_to_2_inst_3/aclk] [get_bd_pins axis_switch_2_to_1_inst_0/aclk] [get_bd_pins axis_switch_2_to_1_inst_1/aclk] [get_bd_pins axis_switch_2_to_1_inst_2/aclk] [get_bd_pins ccl_offload_0/ap_clk] [get_bd_pins cclo_sq_adapter_0/ap_clk] [get_bd_pins cyt_cq_dm_sts_conver_0/ap_clk] [get_bd_pins cyt_cq_dm_sts_conver_1/ap_clk] [get_bd_pins cyt_dma_sq_adapter_0/ap_clk] [get_bd_pins cyt_rdma_arbiter_0/ap_clk] [get_bd_pins hostctrl_0/ap_clk] [get_bd_pins reduce_ops_0/ap_clk] [get_bd_pins rst_ap_clk_0_250M/slowest_sync_clk] [get_bd_pins smartconnect_0/aclk] [get_bd_pins system_ila_0/clk]
  connect_bd_net -net ap_rst_n_0_1 [get_bd_ports ap_rst_n_0] [get_bd_pins axis_data_fifo_6/s_axis_aresetn] [get_bd_pins axis_data_fifo_7/s_axis_aresetn] [get_bd_pins axis_data_fifo_8/s_axis_aresetn] [get_bd_pins axis_data_fifo_9/s_axis_aresetn] [get_bd_pins axis_register_slice_0/aresetn] [get_bd_pins axis_register_slice_1/aresetn] [get_bd_pins axis_register_slice_10/aresetn] [get_bd_pins axis_register_slice_11/aresetn] [get_bd_pins axis_register_slice_12/aresetn] [get_bd_pins axis_register_slice_13/aresetn] [get_bd_pins axis_register_slice_14/aresetn] [get_bd_pins axis_register_slice_7/aresetn] [get_bd_pins axis_register_slice_8/aresetn] [get_bd_pins axis_register_slice_9/aresetn] [get_bd_pins axis_switch_1_to_2_inst_0/aresetn] [get_bd_pins axis_switch_1_to_2_inst_1/aresetn] [get_bd_pins axis_switch_1_to_2_inst_2/aresetn] [get_bd_pins axis_switch_1_to_2_inst_3/aresetn] [get_bd_pins axis_switch_2_to_1_inst_0/aresetn] [get_bd_pins axis_switch_2_to_1_inst_1/aresetn] [get_bd_pins ccl_offload_0/ap_rst_n] [get_bd_pins cclo_sq_adapter_0/ap_rst_n] [get_bd_pins cyt_cq_dm_sts_conver_0/ap_rst_n] [get_bd_pins cyt_cq_dm_sts_conver_1/ap_rst_n] [get_bd_pins cyt_dma_sq_adapter_0/ap_rst_n] [get_bd_pins cyt_rdma_arbiter_0/ap_rst_n] [get_bd_pins hostctrl_0/ap_rst_n] [get_bd_pins reduce_ops_0/ap_rst_n] [get_bd_pins rst_ap_clk_0_250M/ext_reset_in] [get_bd_pins smartconnect_0/aresetn] [get_bd_pins system_ila_0/resetn]
  connect_bd_net -net rst_ap_clk_0_250M_peripheral_aresetn [get_bd_pins axis_data_fifo_0/s_axis_aresetn] [get_bd_pins axis_data_fifo_1/s_axis_aresetn] [get_bd_pins axis_data_fifo_2/s_axis_aresetn] [get_bd_pins axis_data_fifo_3/s_axis_aresetn] [get_bd_pins axis_data_fifo_4/s_axis_aresetn] [get_bd_pins axis_data_fifo_5/s_axis_aresetn] [get_bd_pins axis_register_slice_2/aresetn] [get_bd_pins axis_register_slice_3/aresetn] [get_bd_pins axis_register_slice_4/aresetn] [get_bd_pins axis_register_slice_5/aresetn] [get_bd_pins axis_register_slice_6/aresetn] [get_bd_pins axis_switch_2_to_1_inst_2/aresetn] [get_bd_pins rst_ap_clk_0_250M/peripheral_aresetn]
  connect_bd_net -net xlconstant_0_dout [get_bd_pins axis_switch_2_to_1_inst_0/s_req_suppress] [get_bd_pins xlconstant_0/dout]
  connect_bd_net -net xlconstant_1_dout [get_bd_pins axis_switch_2_to_1_inst_1/s_req_suppress] [get_bd_pins xlconstant_1/dout]
  connect_bd_net -net xlconstant_2_dout [get_bd_pins axis_switch_2_to_1_inst_2/s_req_suppress] [get_bd_pins xlconstant_2/dout]

  # Create address segments
  assign_bd_address -offset 0x00000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs ccl_offload_0/s_axi_control/reg0] -force
  assign_bd_address -offset 0x00002000 -range 0x00002000 -target_address_space [get_bd_addr_spaces S00_AXI_0] [get_bd_addr_segs hostctrl_0/s_axi_control/Reg] -force



validate_bd_design
save_bd_design

make_wrapper -files [get_files "$build_dir/test_config_0/user_c0_0/test.srcs/sources_1/bd/accl_bd/accl_bd.bd"] -top
add_files -norecurse "$build_dir/test_config_0/user_c0_0/test.srcs/sources_1/bd/accl_bd/hdl/accl_bd_wrapper.v"
update_compile_order -fileset sources_1
exit


