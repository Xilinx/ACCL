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

set stacktype [lindex $::argv 0]
set en_dma [lindex $::argv 1]
set en_arith [lindex $::argv 2]
set en_compress [lindex $::argv 3]
set en_extkrnl [lindex $::argv 4]
set mb_debug_level [lindex $::argv 5]

set kernel_name    "ccl_offload"
set kernel_vendor  "Xilinx"
set kernel_library "ACCL"

proc config_axis_if {core ifname clkname data user id dest rdy strb keep last} {
    puts "Setting AXI-Stream attributes for $ifname and associating with $clkname"
    ::ipx::associate_bus_interfaces -busif $ifname -clock $clkname $core
    set axis_bif      [::ipx::get_bus_interfaces -of $core $ifname] 
    set bifparam [::ipx::add_bus_parameter -quiet "TDATA_NUM_BYTES" $axis_bif]
    set_property value        $data   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "TUSER_WIDTH" $axis_bif]
    set_property value        $user   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "TID_WIDTH" $axis_bif]
    set_property value        $id   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "TDEST_WIDTH" $axis_bif]
    set_property value        $dest  $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "HAS_TREADY" $axis_bif]
    set_property value        $rdy   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "HAS_TSTRB" $axis_bif]
    set_property value        $strb   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "HAS_TKEEP" $axis_bif]
    set_property value        $keep   $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "HAS_TLAST" $axis_bif]
    set_property value        $last   $bifparam
    set_property value_source constant     $bifparam
}

proc config_axi_if {core ifname clkname bsize maxr maxw} {
    puts "Setting AXI-MM attributes for $ifname and associating with $clkname"
    ::ipx::associate_bus_interfaces -busif $ifname -clock $clkname $core
    set bif      [::ipx::get_bus_interfaces -of $core $ifname] 
    set bifparam [::ipx::add_bus_parameter -quiet "MAX_BURST_LENGTH" $bif]
    set_property value        $bsize           $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "NUM_READ_OUTSTANDING" $bif]
    set_property value        $maxr           $bifparam
    set_property value_source constant     $bifparam
    set bifparam [::ipx::add_bus_parameter -quiet "NUM_WRITE_OUTSTANDING" $bif]
    set_property value        $maxw           $bifparam
    set_property value_source constant     $bifparam
}

proc config_axilite_reg {addr_block name offset size intf} {
    set reg      [::ipx::add_register -quiet $name $addr_block]
    set_property address_offset $offset $reg
    set_property size           $size   $reg
    if {$intf ne ""} {
        set regparam [::ipx::add_register_parameter -quiet {ASSOCIATED_BUSIF} $reg] 
        set_property value $intf $regparam 
    }
}

proc edit_core {core} {

    global stacktype
    global en_arith
    global en_compress
    global en_dma
    global en_extkrnl
    global mb_debug_level

    puts "Configuring core interfaces"

    ::ipx::associate_bus_interfaces -busif "s_axi_control" -clock "ap_clk" $core

    config_axis_if $core "s_axis_call_req" "ap_clk" 4 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_call_ack" "ap_clk" 4 0 0 0 1 0 1 1

    config_axis_if $core "s_axis_eth_rx_data" "ap_clk" 64 0 0 8 1 0 1 1
    config_axis_if $core "m_axis_eth_tx_data" "ap_clk" 64 0 0 8 1 0 1 1

    if { $en_dma == 1 } {
        config_axis_if $core "m_axis_dma0_s2mm" "ap_clk" 64 0 0 8 1 0 1 1
        config_axis_if $core "s_axis_dma0_mm2s" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_dma1_s2mm" "ap_clk" 64 0 0 8 1 0 1 1
        config_axis_if $core "s_axis_dma1_mm2s" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_dma0_mm2s_cmd" "ap_clk" 13 0 0 0 1 0 0 0
        config_axis_if $core "s_axis_dma0_mm2s_sts" "ap_clk" 1 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_dma0_s2mm_cmd" "ap_clk" 13 0 0 0 1 0 0 0
        config_axis_if $core "s_axis_dma0_s2mm_sts" "ap_clk" 4 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_dma1_mm2s_cmd" "ap_clk" 13 0 0 0 1 0 0 0
        config_axis_if $core "s_axis_dma1_mm2s_sts" "ap_clk" 1 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_dma1_s2mm_cmd" "ap_clk" 13 0 0 0 1 0 0 0
        config_axis_if $core "s_axis_dma1_s2mm_sts" "ap_clk" 4 0 0 0 1 0 1 1
    }

    if { $stacktype == "TCP" } {
        config_axis_if $core "s_axis_eth_notification" "ap_clk" 16 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_eth_read_pkg" "ap_clk" 4 0 0 0 1 0 1 1
        config_axis_if $core "s_axis_eth_rx_meta" "ap_clk" 2 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_eth_tx_meta" "ap_clk" 4 0 0 0 1 0 1 1
        config_axis_if $core "s_axis_eth_tx_status" "ap_clk" 8 0 0 0 1 0 1 1
    }

    if { $stacktype == "RDMA" } {
        config_axis_if $core "m_axis_rdma_sq" "ap_clk" 16 0 0 0 1 0 0 0
        config_axis_if $core "s_axis_eth_notification" "ap_clk" 8 0 0 0 1 0 0 0
    }

    if { $en_extkrnl == 1 } {
        config_axis_if $core "s_axis_krnl" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_krnl" "ap_clk" 64 0 0 8 1 0 1 1
    }

    if { $en_compress == 1 } {
        config_axis_if $core "s_axis_compression0" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_compression0" "ap_clk" 64 0 0 8 1 0 1 1
        config_axis_if $core "s_axis_compression1" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_compression1" "ap_clk" 64 0 0 8 1 0 1 1
        config_axis_if $core "s_axis_compression2" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_compression2" "ap_clk" 64 0 0 8 1 0 1 1
    }

    if { $en_arith == 1 } {
        config_axis_if $core "s_axis_arith_res" "ap_clk" 64 0 0 0 1 0 1 1
        config_axis_if $core "m_axis_arith_op0" "ap_clk" 64 0 0 8 1 0 1 1
        config_axis_if $core "m_axis_arith_op1" "ap_clk" 64 0 0 8 1 0 1 1
    }

    # Specify the freq_hz parameter 
    set clkbif      [::ipx::get_bus_interfaces -of $core "ap_clk"]
    set clkbifparam [::ipx::add_bus_parameter -quiet "FREQ_HZ" $clkbif]
    # Set desired frequency                   
    set_property value 250000000 $clkbifparam
    # set value_resolve_type 'user' if the frequency can vary. 
    set_property value_resolve_type user $clkbifparam
    # set value_resolve_type 'immediate' if the frequency cannot change. 
    # set_property value_resolve_type immediate $clkbifparam
    ::ipx::infer_bus_interfaces "xilinx.com:interface:bscan_rtl:1.0" $core
    ::ipx::remove_bus_interface bscan_0_reset $core
    set mem_map    [::ipx::add_memory_map -quiet "s_axi_control" $core]
    set addr_block [::ipx::add_address_block -quiet "reg0" $mem_map]

    config_axilite_reg $addr_block "buf0_ptr" 0x10 64 "m_axi_0"
    config_axilite_reg $addr_block "buf1_ptr" 0x20 64 "m_axi_1"
    set_property slave_memory_map_ref "s_axi_control" [::ipx::get_bus_interfaces -of $core "s_axi_control"]

    puts "Setting core attributes"
    set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
    set_property sdx_kernel true $core
    set_property sdx_kernel_type rtl $core
    set_property vitis_drc {ctrl_protocol user_managed} $core
    set_property ipi_drc {ignore_freq_hz true} $core
    ::ipx::save_core $core
}

proc package_project_dcp_and_xdc {path_to_dcp path_to_xdc path_to_packaged kernel_vendor kernel_library kernel_name} {
    set core [::ipx::package_checkpoint -dcp_file $path_to_dcp -root_dir $path_to_packaged -vendor $kernel_vendor -library $kernel_library -name $kernel_name -taxonomy "/KernelIP" -force]
    edit_core $core
    set rel_path_to_xdc [file join "impl" [file tail $path_to_xdc]]
    set abs_path_to_xdc [file join $path_to_packaged $rel_path_to_xdc]
    file mkdir [file dirname $abs_path_to_xdc]
    file copy $path_to_xdc $abs_path_to_xdc
    set xdcfile [::ipx::add_file $rel_path_to_xdc [::ipx::add_file_group "xilinx_implementation" $core]]
    set_property type "xdc" $xdcfile
    set_property used_in [list "implementation"] $xdcfile
    ::ipx::update_checksums $core
    ::ipx::check_integrity -kernel $core
    ::ipx::save_core $core
    ::ipx::unload_core $core
    unset core
}

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr

package_project_dcp_and_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc ./ccl_offload_ex/ccl_offload $kernel_vendor $kernel_library $kernel_name
package_xo -f -xo_path [pwd]/ccl_offload.xo -kernel_name ccl_offload -ip_directory ./ccl_offload_ex/ccl_offload -kernel_xml ./ccl_offload.xml

# close and exit
close_project
exit
