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

# Hierarchical cell: microblaze_0_local_memory
proc create_hier_cell_microblaze_0_local_memory { parentCell nameHier } {

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_microblaze_0_local_memory() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode MirroredMaster -vlnv xilinx.com:interface:lmb_rtl:1.0 DLMB

  create_bd_intf_pin -mode MirroredMaster -vlnv xilinx.com:interface:lmb_rtl:1.0 ILMB


  # Create pins
  create_bd_pin -dir I -type clk LMB_Clk
  create_bd_pin -dir I -type rst SYS_Rst

  # Create instance: dlmb_bram_if_cntlr, and set properties
  set dlmb_bram_if_cntlr [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 dlmb_bram_if_cntlr ]
  set_property -dict [ list \
   CONFIG.C_ECC {0} \
 ] $dlmb_bram_if_cntlr

  # Create instance: dlmb_bram_if_cntlr_bram, and set properties
  set dlmb_bram_if_cntlr_bram [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 dlmb_bram_if_cntlr_bram ]
  set_property -dict [ list \
   CONFIG.Memory_Type {True_Dual_Port_RAM} \
 ] $dlmb_bram_if_cntlr_bram

  # Create instance: dlmb_v10, and set properties
  set dlmb_v10 [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 dlmb_v10 ]
  set_property -dict [ list \
   CONFIG.C_LMB_NUM_SLAVES {2} \
 ] $dlmb_v10

  # Create instance: ilmb_bram_if_cntlr, and set properties
  set ilmb_bram_if_cntlr [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 ilmb_bram_if_cntlr ]
  set_property -dict [ list \
   CONFIG.C_ECC {0} \
 ] $ilmb_bram_if_cntlr

  # Create instance: ilmb_v10, and set properties
  set ilmb_v10 [ create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 ilmb_v10 ]

  # Create interface connections
  connect_bd_intf_net -intf_net dlmb_bram_if_cntlr_BRAM_PORT [get_bd_intf_pins dlmb_bram_if_cntlr/BRAM_PORT] [get_bd_intf_pins dlmb_bram_if_cntlr_bram/BRAM_PORTA]
  connect_bd_intf_net -intf_net ilmb_bram_if_cntlr_BRAM_PORT [get_bd_intf_pins dlmb_bram_if_cntlr_bram/BRAM_PORTB] [get_bd_intf_pins ilmb_bram_if_cntlr/BRAM_PORT]
  connect_bd_intf_net -intf_net microblaze_0_dlmb [get_bd_intf_pins DLMB] [get_bd_intf_pins dlmb_v10/LMB_M]
  connect_bd_intf_net -intf_net microblaze_0_dlmb_bus [get_bd_intf_pins dlmb_bram_if_cntlr/SLMB] [get_bd_intf_pins dlmb_v10/LMB_Sl_0]
  connect_bd_intf_net -intf_net microblaze_0_ilmb [get_bd_intf_pins ILMB] [get_bd_intf_pins ilmb_v10/LMB_M]
  connect_bd_intf_net -intf_net microblaze_0_ilmb_bus [get_bd_intf_pins ilmb_bram_if_cntlr/SLMB] [get_bd_intf_pins ilmb_v10/LMB_Sl_0]

  # Create port connections
  connect_bd_net -net microblaze_0_Clk [get_bd_pins LMB_Clk] [get_bd_pins dlmb_bram_if_cntlr/LMB_Clk] [get_bd_pins dlmb_v10/LMB_Clk] [get_bd_pins ilmb_bram_if_cntlr/LMB_Clk] [get_bd_pins ilmb_v10/LMB_Clk]
  connect_bd_net -net microblaze_0_LMB_Rst [get_bd_pins SYS_Rst] [get_bd_pins dlmb_bram_if_cntlr/LMB_Rst] [get_bd_pins dlmb_v10/SYS_Rst] [get_bd_pins ilmb_bram_if_cntlr/LMB_Rst] [get_bd_pins ilmb_v10/SYS_Rst]

  # Restore current instance
  current_bd_instance $oldCurInst
}

proc create_dma_infrastructure { dmaIndex } {
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_mm2s_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_mm2s_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_s2mm_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 dma${dmaIndex}_s2mm_sts

  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_mm2s_cmd
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {13} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma${dmaIndex}_mm2s_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_mm2s_sts
  set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma${dmaIndex}_mm2s_sts]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_s2mm_cmd
  set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {13} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma${dmaIndex}_s2mm_cmd]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma${dmaIndex}_s2mm_sts
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma${dmaIndex}_s2mm_sts]

  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_mm2s_cmd] [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_cmd/M_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_mm2s_sts] [get_bd_intf_pins fifo_dma${dmaIndex}_mm2s_sts/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_s2mm_cmd] [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_cmd/M_AXIS]
  connect_bd_intf_net [get_bd_intf_pins dma${dmaIndex}_s2mm_sts] [get_bd_intf_pins fifo_dma${dmaIndex}_s2mm_sts/S_AXIS]

  connect_bd_net [get_bd_pins proc_sys_reset_1/peripheral_aresetn]  [get_bd_pins fifo_dma${dmaIndex}_mm2s_sts/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_mm2s_cmd/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_s2mm_sts/s_axis_aresetn] \
                                                                    [get_bd_pins fifo_dma${dmaIndex}_s2mm_cmd/s_axis_aresetn] 
  connect_bd_net [get_bd_pins ap_clk] [get_bd_pins fifo_dma${dmaIndex}_mm2s_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_mm2s_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_s2mm_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_dma${dmaIndex}_s2mm_cmd/s_axis_aclk]
}

# Command arbiter
proc create_hier_cell_cmd_arbiter { parentCell nameHier {numCmdStreams 1} } {

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_cmd_arbiter() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: control
proc create_hier_cell_control { parentCell nameHier {mbDebugLevel 0} {fanInSupport 0} } {

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_control() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 encore_control
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 host_control

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 eth_packetizer_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 eth_depacketizer_sts
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 eth_packetizer_sts
  
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 eth_opencon_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 eth_opencon_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 eth_openport_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 eth_openport_sts

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 call_req
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 call_ack

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma0_rd_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 dma1_rd_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 krnl_in_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 krnl_out_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 arith_op0_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 arith_op1_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 arith_res_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane0_op_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane0_res_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane1_op_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane1_res_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane2_op_seg_cmd
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 clane2_res_seg_cmd
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 krnl_out_seg_sts

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 m_axi_cfg

  # Create pins
  create_bd_pin -dir I -type clk ap_clk
  create_bd_pin -dir I -type rst ap_rst_n
  create_bd_pin -dir O -from 0 -to 0 -type rst encore_aresetn

  # Create instance: fifo_eth_packetizer_cmd, and set properties
  set fifo_eth_packetizer_cmd [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_eth_packetizer_cmd ]
  set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {24} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] $fifo_eth_packetizer_cmd
  # Create instance: fifo_eth_depacketizer_sts, and set properties
  set fifo_eth_depacketizer_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_eth_depacketizer_sts ]
  set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {24} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] $fifo_eth_depacketizer_sts
   # Create instance: fifo_eth_packetizer_sts, and set properties
  set fifo_eth_packetizer_sts [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_eth_packetizer_sts ]
  set_property -dict [ list  CONFIG.HAS_TLAST {1}  CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] $fifo_eth_packetizer_sts
  # Create instance: tcp session handler
  set tcp_session [create_bd_cell -type ip -vlnv xilinx.com:hls:tcp_sessionHandler:1.0 tcp_session]

  # Create instance: microblaze_0, and set properties
  # TODO: make debug/trace optional; the profiling buffer is large
  # TODO: enable caches and/or hw support for mul/div
  set microblaze_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze:11.0 microblaze_0 ]
  set_property -dict [ list \
   CONFIG.C_BASE_VECTORS {0x0000000000010000} \
   CONFIG.C_USE_INTERRUPT {0} \
   CONFIG.C_D_AXI {1} \
   CONFIG.C_D_LMB {1} \
   CONFIG.C_FSL_LINKS {4} \
   CONFIG.C_I_LMB {1} \
   CONFIG.C_TRACE {0} \
   CONFIG.C_USE_EXTENDED_FSL_INSTR {1} \
   CONFIG.C_USE_MSR_INSTR {1} \
   CONFIG.C_USE_PCMP_INSTR {1} \
   CONFIG.C_DEBUG_EXTERNAL_TRACE.VALUE_SRC PROPAGATED \
   CONFIG.C_DEBUG_ENABLED {0} \
   CONFIG.C_USE_BARREL {1} \
   CONFIG.C_USE_DIV {1} \
   CONFIG.C_USE_HW_MUL {1} \
   CONFIG.C_ADDR_TAG_BITS {0} \
   CONFIG.C_DCACHE_ADDR_TAG {0} \
   CONFIG.C_USE_BRANCH_TARGET_CACHE {1} \
   CONFIG.C_BRANCH_TARGET_CACHE_SIZE {5} \
 ] $microblaze_0

  # Create instance: axi_gpio_0, and set properties
  set gpio [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0 ]
  set_property -dict [ list \
   CONFIG.C_IS_DUAL {1} \
   CONFIG.C_ALL_INPUTS {0} \
   CONFIG.C_ALL_OUTPUTS {1} \
   CONFIG.C_ALL_INPUTS_2 {1} \
   CONFIG.C_GPIO2_WIDTH {32} \
   CONFIG.C_GPIO_WIDTH {2} \
   CONFIG.C_INTERRUPT_PRESENT {0} \
 ] $gpio

  # Create instance: xlconstant_hwid, and set properties
  set hwid [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_hwid ]
  set_property -dict [ list \
   CONFIG.CONST_WIDTH {32} \
   CONFIG.CONST_VAL {0xcafebabe} \
 ] $hwid

  # Create instance: xlslice_encore_rstn, and set properties
  set slice_encore_rstn [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_encore_rstn ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {1} \
   CONFIG.DIN_TO {1} \
   CONFIG.DIN_WIDTH {3} \
   CONFIG.DOUT_WIDTH {1} \
 ] $slice_encore_rstn

  # connect GPIO to constant and to encore reset slicer
  connect_bd_net [get_bd_pins $hwid/dout] [get_bd_pins $gpio/gpio2_io_i]
  connect_bd_net [get_bd_pins $gpio/gpio_io_o] [get_bd_pins $slice_encore_rstn/Din]

  # Create crossbar for joint access to config memory (usually in PLRAM)
  set exchmem_xbar [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 exchmem_xbar ]
  set_property -dict [ list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2} CONFIG.R_REGISTER {1} ] $exchmem_xbar
  connect_bd_intf_net [get_bd_intf_pins $exchmem_xbar/M00_AXI] [get_bd_intf_pins m_axi_cfg]

  # Create register_slice for host access to GPIO
  set host_rslice [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_register_slice:2.1 host_rslice ]
  set_property -dict [ list CONFIG.REG_B {7} CONFIG.REG_R {7} CONFIG.REG_W {7} ] $host_rslice

  # Create crossbar for joint access to GPIO
  set gpio_xbar [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 gpio_xbar ]
  set_property -dict [ list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2} CONFIG.R_REGISTER {1} ] $gpio_xbar

 if { $mbDebugLevel != 0 } {
   # Create instance: mdm_1, and set properties
   set mdm_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:mdm:3.2 mdm_1 ]
   set_property -dict [ list CONFIG.C_USE_BSCAN {2} ] $mdm_1
   # Enable debug and trace on Microblaze
   set_property -dict [ list CONFIG.C_TRACE {1} CONFIG.C_DEBUG_ENABLED $mbDebugLevel ] $microblaze_0
   # Create ports and connect nets
   create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:mbtrace_rtl:2.0 TRACE
   create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:bscan_rtl:1.0 bscan_0
   connect_bd_intf_net -intf_net bscan_0 [get_bd_intf_pins bscan_0] [get_bd_intf_pins mdm_1/BSCAN]
   connect_bd_intf_net -intf_net microblaze_0_TRACE [get_bd_intf_pins TRACE] [get_bd_intf_pins microblaze_0/TRACE]
   connect_bd_intf_net -intf_net microblaze_0_debug [get_bd_intf_pins mdm_1/MBDEBUG_0] [get_bd_intf_pins microblaze_0/DEBUG]
 }

  # Create instance: mb_periph_xbar, and set properties
  set mb_periph_xbar [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 mb_periph_xbar ]
  set_property -dict [ list CONFIG.NUM_MI {6} ] $mb_periph_xbar

  # Create instance: microblaze_0_local_memory
  create_hier_cell_microblaze_0_local_memory $hier_obj microblaze_0_local_memory

  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_1 ]

  # Create DMAs
  create_dma_infrastructure 0
  create_dma_infrastructure 1

  # Create DMA enqueue/dequeue blocks
  set rxbuf_enqueue [create_bd_cell -type ip -vlnv xilinx.com:hls:rxbuf_enqueue:1.0 rxbuf_enqueue]
  set rxbuf_dequeue [create_bd_cell -type ip -vlnv xilinx.com:hls:rxbuf_dequeue:1.0 rxbuf_dequeue]
  set rxbuf_seek [create_bd_cell -type ip -vlnv xilinx.com:hls:rxbuf_seek:1.0 rxbuf_seek]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_seek
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {20} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_seek]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_inflight
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_inflight]

  connect_bd_intf_net [get_bd_intf_pins rxbuf_enqueue/s_axi_control] [get_bd_intf_pins $mb_periph_xbar/M02_AXI]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_dequeue/s_axi_control] [get_bd_intf_pins $mb_periph_xbar/M03_AXI]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_seek/s_axi_control] [get_bd_intf_pins $mb_periph_xbar/M04_AXI]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_dequeue/notification_queue] [get_bd_intf_pins fifo_seek/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_seek/M_AXIS] [get_bd_intf_pins rxbuf_seek/rx_notify]

  if { $fanInSupport == 1 } {

    create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 eth_depacketizer_notif

    create_bd_cell -type ip -vlnv xilinx.com:hls:rxbuf_session:1.0 rxbuf_session
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_inflight_session
    set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_inflight_session]

    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dmacmd_session
    set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {13} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dmacmd_session]
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dmasts_session
    set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dmasts_session]

    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_hdr_session
    set_property -dict [ list CONFIG.HAS_TLAST {1} CONFIG.TDATA_NUM_BYTES {24} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_hdr_session]
  

    connect_bd_intf_net [get_bd_intf_pins rxbuf_enqueue/dma_cmd] [get_bd_intf_pins fifo_dmacmd_session/S_AXIS] 
    connect_bd_intf_net [get_bd_intf_pins fifo_dmacmd_session/M_AXIS] [get_bd_intf_pins rxbuf_session/rxbuf_dma_cmd]

    connect_bd_intf_net [get_bd_intf_pins rxbuf_session/rxbuf_dma_sts] [get_bd_intf_pins fifo_dmasts_session/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_dmasts_session/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/dma_sts]

    connect_bd_intf_net [get_bd_intf_pins rxbuf_session/fragment_dma_cmd] [get_bd_intf_pins fifo_dma0_s2mm_cmd/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_dma0_s2mm_sts/M_AXIS] [get_bd_intf_pins rxbuf_session/fragment_dma_sts]

    connect_bd_intf_net [get_bd_intf_pins rxbuf_enqueue/inflight_queue] [get_bd_intf_pins fifo_inflight_session/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_inflight_session/M_AXIS] [get_bd_intf_pins rxbuf_session/rxbuf_idx_in]

    connect_bd_intf_net [get_bd_intf_pins rxbuf_session/rxbuf_idx_out] [get_bd_intf_pins fifo_inflight/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_inflight/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/inflight_queue]

    connect_bd_intf_net [get_bd_intf_pins fifo_eth_depacketizer_sts/M_AXIS] [get_bd_intf_pins rxbuf_session/eth_hdr_in]

    connect_bd_intf_net [get_bd_intf_pins rxbuf_session/eth_hdr_out] [get_bd_intf_pins fifo_hdr_session/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_hdr_session/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/eth_hdr]

    connect_bd_intf_net [get_bd_intf_pins eth_depacketizer_notif] [get_bd_intf_pins rxbuf_session/session_notification]

    connect_bd_net [get_bd_pins ap_clk] [get_bd_pins fifo_inflight_session/s_axis_aclk] \
                                        [get_bd_pins fifo_dmacmd_session/s_axis_aclk] \
                                        [get_bd_pins fifo_dmasts_session/s_axis_aclk] \
                                        [get_bd_pins fifo_hdr_session/s_axis_aclk] \
                                        [get_bd_pins rxbuf_session/ap_clk]

    connect_bd_net [get_bd_pins proc_sys_reset_1/peripheral_aresetn] \
                    [get_bd_pins fifo_inflight_session/s_axis_aresetn] \
                    [get_bd_pins fifo_dmacmd_session/s_axis_aresetn] \
                    [get_bd_pins fifo_dmasts_session/s_axis_aresetn] \
                    [get_bd_pins fifo_hdr_session/s_axis_aresetn] \
                    [get_bd_pins rxbuf_session/ap_rst_n]

  } else {

    connect_bd_intf_net [get_bd_intf_pins fifo_eth_depacketizer_sts/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/eth_hdr]
    connect_bd_intf_net [get_bd_intf_pins fifo_dma0_s2mm_cmd/S_AXIS] [get_bd_intf_pins rxbuf_enqueue/dma_cmd]
    connect_bd_intf_net [get_bd_intf_pins fifo_dma0_s2mm_sts/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/dma_sts]
    connect_bd_intf_net [get_bd_intf_pins rxbuf_enqueue/inflight_queue] [get_bd_intf_pins fifo_inflight/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_inflight/M_AXIS] [get_bd_intf_pins rxbuf_dequeue/inflight_queue]

  }


  # Create DMA segmentation processor
  set dma_mover [ create_bd_cell -type ip -vlnv xilinx.com:hls:dma_mover:1.0 dma_mover ]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma_mover_command
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma_mover_command]
  create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_dma_mover_error
  set_property -dict [ list CONFIG.HAS_TLAST {0} CONFIG.TDATA_NUM_BYTES {4} CONFIG.FIFO_DEPTH {32} CONFIG.FIFO_MEMORY_TYPE {distributed}] [get_bd_cells fifo_dma_mover_error]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma0_mm2s_cmd/S_AXIS] [get_bd_intf_pins dma_mover/dma0_read_cmd]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma0_mm2s_sts/M_AXIS] [get_bd_intf_pins dma_mover/dma0_read_sts]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma1_mm2s_cmd/S_AXIS] [get_bd_intf_pins dma_mover/dma1_read_cmd]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma1_mm2s_sts/M_AXIS] [get_bd_intf_pins dma_mover/dma1_read_sts]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma1_s2mm_cmd/S_AXIS] [get_bd_intf_pins dma_mover/dma1_write_cmd]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma1_s2mm_sts/M_AXIS] [get_bd_intf_pins dma_mover/dma1_write_sts]
  connect_bd_intf_net [get_bd_intf_pins fifo_eth_packetizer_cmd/S_AXIS] [get_bd_intf_pins dma_mover/eth_cmd]
  connect_bd_intf_net [get_bd_intf_pins fifo_eth_packetizer_sts/M_AXIS] [get_bd_intf_pins dma_mover/eth_sts]

  #interconnect to access exchange memory from offload cores
  set dma_memory_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_crossbar:2.1 dma_memory_ic]
  set_property -dict [list CONFIG.NUM_SI {4} CONFIG.NUM_MI {1}] $dma_memory_ic
  connect_bd_intf_net [get_bd_intf_pins rxbuf_enqueue/m_axi_mem] [get_bd_intf_pins dma_memory_ic/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_dequeue/m_axi_mem] [get_bd_intf_pins dma_memory_ic/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_seek/m_axi_mem] [get_bd_intf_pins dma_memory_ic/S02_AXI]
  connect_bd_intf_net [get_bd_intf_pins dma_mover/m_axi_mem] [get_bd_intf_pins dma_memory_ic/S03_AXI]
  connect_bd_intf_net [get_bd_intf_pins dma_memory_ic/M00_AXI] [get_bd_intf_pins $exchmem_xbar/S01_AXI]
  
  # Create interface connections
  connect_bd_intf_net -intf_net microblaze_0_dlmb_1 [get_bd_intf_pins microblaze_0/DLMB] [get_bd_intf_pins microblaze_0_local_memory/DLMB]
  connect_bd_intf_net -intf_net microblaze_0_ilmb_1 [get_bd_intf_pins microblaze_0/ILMB] [get_bd_intf_pins microblaze_0_local_memory/ILMB]

  connect_bd_intf_net [get_bd_intf_pins $mb_periph_xbar/M00_AXI] [get_bd_intf_pins $exchmem_xbar/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins $mb_periph_xbar/M01_AXI] [get_bd_intf_pins encore_control]

  connect_bd_intf_net [get_bd_intf_pins host_control] [get_bd_intf_pins $host_rslice/S_AXI]
  connect_bd_intf_net [get_bd_intf_pins $host_rslice/M_AXI] [get_bd_intf_pins $gpio_xbar/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins $mb_periph_xbar/M05_AXI] [get_bd_intf_pins $gpio_xbar/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins $gpio_xbar/M00_AXI] [get_bd_intf_pins $gpio/S_AXI]

  connect_bd_intf_net -intf_net eth_openport_cmd [get_bd_intf_pins tcp_session/m_axis_tcp_listen_port] [get_bd_intf_pins eth_openport_cmd]
  connect_bd_intf_net -intf_net eth_opencon_cmd [get_bd_intf_pins tcp_session/m_axis_tcp_open_connection] [get_bd_intf_pins eth_opencon_cmd]
  connect_bd_intf_net -intf_net eth_packetizer_cmd [get_bd_intf_pins fifo_eth_packetizer_cmd/M_AXIS] [get_bd_intf_pins eth_packetizer_cmd]
  connect_bd_intf_net -intf_net eth_openport_sts [get_bd_intf_pins tcp_session/s_axis_tcp_port_status] [get_bd_intf_pins eth_openport_sts]
  connect_bd_intf_net -intf_net eth_opencon_sts [get_bd_intf_pins tcp_session/s_axis_tcp_open_status] [get_bd_intf_pins eth_opencon_sts]
  connect_bd_intf_net -intf_net eth_packetizer_sts [get_bd_intf_pins fifo_eth_packetizer_sts/S_AXIS] [get_bd_intf_pins eth_packetizer_sts]
  connect_bd_intf_net [get_bd_intf_pins fifo_eth_depacketizer_sts/S_AXIS] [get_bd_intf_pins eth_depacketizer_sts]

  connect_bd_intf_net [get_bd_intf_pins microblaze_0/M_AXI_DP] [get_bd_intf_pins $mb_periph_xbar/S00_AXI]

  connect_bd_intf_net [get_bd_intf_pins tcp_session/port_cmd] [get_bd_intf_pins microblaze_0/M2_AXIS]
  connect_bd_intf_net [get_bd_intf_pins tcp_session/port_sts] [get_bd_intf_pins microblaze_0/S2_AXIS]

  connect_bd_intf_net [get_bd_intf_pins tcp_session/con_cmd] [get_bd_intf_pins microblaze_0/M3_AXIS]
  connect_bd_intf_net [get_bd_intf_pins tcp_session/con_sts] [get_bd_intf_pins microblaze_0/S3_AXIS]

  connect_bd_intf_net [get_bd_intf_pins call_req] [get_bd_intf_pins microblaze_0/S0_AXIS]
  connect_bd_intf_net [get_bd_intf_pins call_ack] [get_bd_intf_pins microblaze_0/M0_AXIS]

  connect_bd_intf_net [get_bd_intf_pins dma_mover/error] [get_bd_intf_pins fifo_dma_mover_error/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma_mover_error/M_AXIS] [get_bd_intf_pins microblaze_0/S1_AXIS]
  connect_bd_intf_net [get_bd_intf_pins microblaze_0/M1_AXIS] [get_bd_intf_pins fifo_dma_mover_command/S_AXIS]
  connect_bd_intf_net [get_bd_intf_pins fifo_dma_mover_command/M_AXIS] [get_bd_intf_pins dma_mover/command]

  connect_bd_intf_net [get_bd_intf_pins dma_mover/rxbuf_req] [get_bd_intf_pins rxbuf_seek/rx_seek_request]
  connect_bd_intf_net [get_bd_intf_pins dma_mover/rxbuf_release_req] [get_bd_intf_pins rxbuf_seek/rx_release_request]
  connect_bd_intf_net [get_bd_intf_pins rxbuf_seek/rx_seek_ack] [get_bd_intf_pins dma_mover/rxbuf_ack]

  connect_bd_intf_net [get_bd_intf_pins dma0_rd_seg_cmd] [get_bd_intf_pins dma_mover/dma0_read_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins dma1_rd_seg_cmd] [get_bd_intf_pins dma_mover/dma1_read_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins krnl_in_seg_cmd] [get_bd_intf_pins dma_mover/krnl_in_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins krnl_out_seg_cmd] [get_bd_intf_pins dma_mover/krnl_out_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins arith_op0_seg_cmd] [get_bd_intf_pins dma_mover/arith_op0_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins arith_op1_seg_cmd] [get_bd_intf_pins dma_mover/arith_op1_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins arith_res_seg_cmd] [get_bd_intf_pins dma_mover/arith_res_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane0_op_seg_cmd] [get_bd_intf_pins dma_mover/clane0_op_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane0_res_seg_cmd] [get_bd_intf_pins dma_mover/clane0_res_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane1_op_seg_cmd] [get_bd_intf_pins dma_mover/clane1_op_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane1_res_seg_cmd] [get_bd_intf_pins dma_mover/clane1_res_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane2_op_seg_cmd] [get_bd_intf_pins dma_mover/clane2_op_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins clane2_res_seg_cmd] [get_bd_intf_pins dma_mover/clane2_res_seg_cmd]
  connect_bd_intf_net [get_bd_intf_pins krnl_out_seg_sts] [get_bd_intf_pins dma_mover/krnl_out_seg_sts]

  # Clocks and resets
  connect_bd_net -net SYS_Rst_1 [get_bd_pins microblaze_0_local_memory/SYS_Rst] [get_bd_pins proc_sys_reset_0/peripheral_reset]
  connect_bd_net [get_bd_pins ap_clk] [get_bd_pins tcp_session/ap_clk] \
                                      [get_bd_pins fifo_eth_depacketizer_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_eth_packetizer_cmd/s_axis_aclk] \
                                      [get_bd_pins fifo_eth_packetizer_sts/s_axis_aclk] \
                                      [get_bd_pins fifo_seek/s_axis_aclk] \
                                      [get_bd_pins fifo_inflight/s_axis_aclk] \
                                      [get_bd_pins microblaze_0/Clk] \
                                      [get_bd_pins $mb_periph_xbar/ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M00_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M01_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M02_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M03_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M04_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/M05_ACLK] \
                                      [get_bd_pins $mb_periph_xbar/S00_ACLK] \
                                      [get_bd_pins $gpio/s_axi_aclk] \
                                      [get_bd_pins $exchmem_xbar/aclk] \
                                      [get_bd_pins $gpio_xbar/aclk] \
                                      [get_bd_pins $host_rslice/aclk] \
                                      [get_bd_pins microblaze_0_local_memory/LMB_Clk] \
                                      [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
                                      [get_bd_pins proc_sys_reset_1/slowest_sync_clk] \
                                      [get_bd_pins rxbuf_enqueue/ap_clk] \
                                      [get_bd_pins rxbuf_dequeue/ap_clk] \
                                      [get_bd_pins rxbuf_seek/ap_clk] \
                                      [get_bd_pins dma_mover/ap_clk] \
                                      [get_bd_pins fifo_dma_mover_command/s_axis_aclk] \
                                      [get_bd_pins fifo_dma_mover_error/s_axis_aclk] \
                                      [get_bd_pins dma_memory_ic/aclk]
                                      
  connect_bd_net [get_bd_pins ap_rst_n] [get_bd_pins proc_sys_reset_0/ext_reset_in]
  connect_bd_net  [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins $mb_periph_xbar/ARESETN] \
                                                                    [get_bd_pins $mb_periph_xbar/M00_ARESETN] \
                                                                    [get_bd_pins $mb_periph_xbar/S00_ARESETN] \
                                                                    [get_bd_pins $mb_periph_xbar/M05_ARESETN] \
                                                                    [get_bd_pins $exchmem_xbar/aresetn] \
                                                                    [get_bd_pins $gpio_xbar/aresetn] \
                                                                    [get_bd_pins $host_rslice/aresetn] \
                                                                    [get_bd_pins $gpio/s_axi_aresetn]

  if { $mbDebugLevel != 0 } {
    connect_bd_net -net mdm_1_Debug_SYS_Rst [get_bd_pins mdm_1/Debug_SYS_Rst] [get_bd_pins proc_sys_reset_0/mb_debug_sys_rst] [get_bd_pins proc_sys_reset_1/mb_debug_sys_rst]
  }
  
  connect_bd_net [get_bd_pins $slice_encore_rstn/Dout] [get_bd_pins proc_sys_reset_1/ext_reset_in]
  connect_bd_net [get_bd_pins microblaze_0/Reset] [get_bd_pins proc_sys_reset_0/mb_reset]
  connect_bd_net [get_bd_pins proc_sys_reset_1/peripheral_aresetn] [get_bd_pins encore_aresetn] \
                                                                   [get_bd_pins $mb_periph_xbar/M01_ARESETN] \
                                                                   [get_bd_pins $mb_periph_xbar/M02_ARESETN] \
                                                                   [get_bd_pins $mb_periph_xbar/M03_ARESETN] \
                                                                   [get_bd_pins $mb_periph_xbar/M04_ARESETN] \
                                                                   [get_bd_pins fifo_eth_depacketizer_sts/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_eth_packetizer_cmd/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_eth_packetizer_sts/s_axis_aresetn] \
                                                                   [get_bd_pins tcp_session/ap_rst_n] \
                                                                   [get_bd_pins fifo_seek/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_inflight/s_axis_aresetn] \
                                                                   [get_bd_pins rxbuf_enqueue/ap_rst_n] \
                                                                   [get_bd_pins rxbuf_dequeue/ap_rst_n] \
                                                                   [get_bd_pins rxbuf_seek/ap_rst_n] \
                                                                   [get_bd_pins dma_mover/ap_rst_n] \
                                                                   [get_bd_pins fifo_dma_mover_command/s_axis_aresetn] \
                                                                   [get_bd_pins fifo_dma_mover_error/s_axis_aresetn] \
                                                                   [get_bd_pins dma_memory_ic/aresetn]

  # Create some hierarchies to keep things organized
  group_bd_cells rxbuf_offload  [get_bd_cells rxbuf_*] \
                                [get_bd_cells fifo_seek] \
                                [get_bd_cells fifo_inflight] \
                                [get_bd_cells fifo_*_session] \
                                [get_bd_cells fifo_eth_depacketizer_sts] \
                                [get_bd_cells fifo_dma0_s2mm_sts] \
                                [get_bd_cells fifo_dma0_s2mm_cmd]

  group_bd_cells dma_offload  [get_bd_cells fifo_eth_packetizer_sts] \
                              [get_bd_cells fifo_dma1_mm2s_sts] \
                              [get_bd_cells fifo_dma1_mm2s_cmd] \
                              [get_bd_cells dma_mover] \
                              [get_bd_cells fifo_dma_mover_command] \
                              [get_bd_cells fifo_dma_mover_error] \
                              [get_bd_cells fifo_dma0_mm2s_cmd] \
                              [get_bd_cells fifo_eth_packetizer_cmd] \
                              [get_bd_cells fifo_dma1_s2mm_sts] \
                              [get_bd_cells fifo_dma0_mm2s_sts] \
                              [get_bd_cells fifo_dma1_s2mm_cmd]

  # Restore current instance
  current_bd_instance $oldCurInst
}
