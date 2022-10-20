## Get variables
if { $::argc != 3 } {
    puts "ERROR: Program \"$::argv0\" requires 3 arguments!, (${argc} given)\n"
    puts "Usage: $::argv0 <xoname> <krnl_name> <device>\n"
    exit
}

set xoname  [lindex $::argv 0]
set krnl_name [lindex $::argv 1]
set device    [lindex $::argv 2]

set suffix "${krnl_name}_${device}"

puts "INFO: xoname-> ${xoname}\n      krnl_name-> ${krnl_name}\n      device-> ${device}\n"

set projName kernel_pack
set path_to_hdl "./src"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_${suffix}"


# set words [split $device "_"]
# set board [lindex $words 1]

# if {[string first "u50" ${board}] != -1} {
#     set projPart "xcu50-fsvh2104-2L-e"
# } elseif {[string first "u55" ${board}] != -1} {
#     set projPart "xcu55c-fsvh2892-2L-e"
# } elseif {[string first "u200" ${board}] != -1} {
#     set projPart "xcu200-fsgd2104-2-e"
# } elseif {[string first "u250" ${board}] != -1} {
#     set projPart "xcu250-figd2104-2L-e"
# } elseif {[string first "u280" ${board}] != -1} {
#     set projPart "xcu280-fsvh2892-2L-e"
# } elseif {[string first "vck5000" ${board}] != -1} {
#     set projPart "xcvc1902-vsva2197-2MP-e-S"
# } else {
#     catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "unsupported device: ${device}"}
#     return 1
# }

set projPart $device

## Create Vivado project and add IP cores
create_project -force $projName $path_to_tmp_project -part $projPart

add_files -norecurse ${path_to_hdl}

set_property top hw_bench [current_fileset]
update_compile_order -fileset sources_1

# Package IP

ipx::package_project -root_dir ${path_to_packaged} -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core ${path_to_packaged}/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory ${path_to_packaged} ${path_to_packaged}/component.xml
set_property core_revision 1 [ipx::current_core]
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::add_bus_interface ap_clk [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:signal:clock:1.0 [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]
ipx::add_port_map CLK [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]
set_property physical_name ap_clk [ipx::get_port_maps CLK -of_objects [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]]
ipx::associate_bus_interfaces -busif cmdIn -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif cmdOut1 -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif cmdOut2 -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif cmdTimestamp -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif stsIn -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif stsOut1 -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif stsOut2 -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif stsTimestamp -clock ap_clk [ipx::current_core]

set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete

## Generate XO
if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ./packaged_kernel_${suffix} -kernel_xml ./kernel.xml



