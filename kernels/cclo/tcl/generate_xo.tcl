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

proc config_axi_if {core ifname bsize maxr maxw} {
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

    config_axi_if $core "m_axi_0" 64 32 32
    config_axi_if $core "m_axi_1" 64 32 32

    ::ipx::associate_bus_interfaces -busif "m_axi_0" -clock "ap_clk" $core
    ::ipx::associate_bus_interfaces -busif "m_axi_1" -clock "ap_clk" $core
    ::ipx::associate_bus_interfaces -busif "s_axi_control" -clock "ap_clk" $core

    config_axis_if $core "s_axis_udp_rx_data" "ap_clk" 64 0 0 16 1 0 1 1
    config_axis_if $core "m_axis_udp_tx_data" "ap_clk" 64 0 0 16 1 0 1 1

    config_axis_if $core "s_axis_tcp_notification" "ap_clk" 16 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_tcp_read_pkg" "ap_clk" 4 0 0 0 1 0 1 1
    config_axis_if $core "s_axis_tcp_rx_meta" "ap_clk" 2 0 0 0 1 0 1 1
    config_axis_if $core "s_axis_tcp_rx_data" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_tcp_tx_meta" "ap_clk" 4 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_tcp_tx_data" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "s_axis_tcp_tx_status" "ap_clk" 8 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_tcp_open_connection" "ap_clk" 8 0 0 0 1 0 1 1
    config_axis_if $core "s_axis_tcp_open_status" "ap_clk" 16 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_tcp_listen_port" "ap_clk" 2 0 0 0 1 0 1 1
    config_axis_if $core "s_axis_tcp_port_status" "ap_clk" 1 0 0 0 1 0 1 1

    config_axis_if $core "s_axis_krnl" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_krnl" "ap_clk" 64 0 0 4 1 0 1 1

    config_axis_if $core "s_axis_compression0" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_compression0" "ap_clk" 64 0 0 4 1 0 1 1
    config_axis_if $core "s_axis_compression1" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_compression1" "ap_clk" 64 0 0 4 1 0 1 1
    config_axis_if $core "s_axis_compression2" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_compression2" "ap_clk" 64 0 0 4 1 0 1 1

    config_axis_if $core "s_axis_arith_res" "ap_clk" 64 0 0 0 1 0 1 1
    config_axis_if $core "m_axis_arith_op" "ap_clk" 128 0 0 4 1 0 1 1

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

    set reg      [::ipx::add_register "CTRL" $addr_block]
    set_property description    "Control signals"    $reg
    set_property address_offset 0x000 $reg
    set_property size           32    $reg
    set field [ipx::add_field AP_START $reg]
        set_property ACCESS {read-write} $field
        set_property BIT_OFFSET {0} $field
        set_property BIT_WIDTH {1} $field
        set_property DESCRIPTION {Control signal Register for 'ap_start'.} $field
        set_property MODIFIED_WRITE_VALUE {modify} $field
    set field [ipx::add_field AP_DONE $reg]
        set_property ACCESS {read-only} $field
        set_property BIT_OFFSET {1} $field
        set_property BIT_WIDTH {1} $field
        set_property DESCRIPTION {Control signal Register for 'ap_done'.} $field
        set_property READ_ACTION {modify} $field
    set field [ipx::add_field AP_IDLE $reg]
        set_property ACCESS {read-only} $field
        set_property BIT_OFFSET {2} $field
        set_property BIT_WIDTH {1} $field
        set_property DESCRIPTION {Control signal Register for 'ap_idle'.} $field
        set_property READ_ACTION {modify} $field
    set field [ipx::add_field AP_READY $reg]
        set_property ACCESS {read-only} $field
        set_property BIT_OFFSET {3} $field
        set_property BIT_WIDTH {1} $field
        set_property DESCRIPTION {Control signal Register for 'ap_ready'.} $field
        set_property READ_ACTION {modify} $field
    set field [ipx::add_field RESERVED_1 $reg]
        set_property ACCESS {read-only} $field
        set_property BIT_OFFSET {4} $field
        set_property BIT_WIDTH {3} $field
        set_property DESCRIPTION {Reserved.  0s on read.} $field
        set_property READ_ACTION {modify} $field
    set field [ipx::add_field AUTO_RESTART $reg]
        set_property ACCESS {read-write} $field
        set_property BIT_OFFSET {7} $field
        set_property BIT_WIDTH {1} $field
        set_property DESCRIPTION {Control signal Register for 'auto_restart'.} $field
        set_property MODIFIED_WRITE_VALUE {modify} $field
    set field [ipx::add_field RESERVED_2 $reg]
        set_property ACCESS {read-only} $field
        set_property BIT_OFFSET {8} $field
        set_property BIT_WIDTH {24} $field
        set_property DESCRIPTION {Reserved.  0s on read.} $field
        set_property READ_ACTION {modify} $field

    config_axilite_reg $addr_block "call_type"          0x10 32 ""
    config_axilite_reg $addr_block "byte_count"         0x18 32 ""
    config_axilite_reg $addr_block "comm"               0x20 32 ""
    config_axilite_reg $addr_block "root_src_dst"       0x28 32 ""
    config_axilite_reg $addr_block "reduce_op"          0x30 32 ""
    config_axilite_reg $addr_block "tag"                0x38 32 ""
    config_axilite_reg $addr_block "datapath_cfg"       0x40 32 ""
    config_axilite_reg $addr_block "compression_flags"  0x48 32 ""
    config_axilite_reg $addr_block "stream_flags"       0x50 32 ""
    config_axilite_reg $addr_block "buf0_ptr"           0x58 64 "m_axi_0"
    config_axilite_reg $addr_block "buf1_ptr"           0x64 64 "m_axi_1"
    config_axilite_reg $addr_block "buf2_ptr"           0x70 64 "m_axi_2"

    set_property slave_memory_map_ref "s_axi_control" [::ipx::get_bus_interfaces -of $core "s_axi_control"]

    set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
    set_property sdx_kernel true $core
    set_property sdx_kernel_type rtl $core
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
    ::ipx::check_integrity -xrt $core
    ::ipx::save_core $core
    ::ipx::unload_core $core
    unset core
}

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr

#run kernel packaging
reset_run synth_1
set extra_synth_options "-mode out_of_context"
if { $en_arith == 1 } { set extra_synth_options "$extra_synth_options -verilog_define ARITH_ENABLE " }
if { $en_compress == 1 } { set extra_synth_options "$extra_synth_options -verilog_define COMPRESSION_ENABLE " }
if { $en_dma == 1 } { set extra_synth_options "$extra_synth_options -verilog_define DMA_ENABLE " }
if { $en_extkrnl == 1 } { set extra_synth_options "$extra_synth_options -verilog_define STREAM_ENABLE " }
if { $stacktype == "TCP" } { set extra_synth_options "$extra_synth_options -verilog_define TCP_ENABLE " }
if { $mb_debug_level > 0 } { set extra_synth_options "$extra_synth_options -verilog_define MB_DEBUG_ENABLE " }
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value $extra_synth_options -objects [get_runs synth_1]
launch_runs synth_1 -jobs 12
wait_on_run [get_runs synth_1]
open_run synth_1 -name synth_1
rename_ref -prefix_all ccl_offload_
write_checkpoint ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
write_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc
close_design
package_project_dcp_and_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc ./ccl_offload_ex/ccl_offload $kernel_vendor $kernel_library $kernel_name
package_xo  -xo_path [pwd]/ccl_offload.xo -kernel_name ccl_offload -ip_directory ./ccl_offload_ex/ccl_offload -kernel_xml ./xml/cclo.xml

# close and exit
close_project
exit
