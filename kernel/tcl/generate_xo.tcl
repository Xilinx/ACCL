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

set elf [lindex $::argv 0]

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

# replace the default top with the modified wrapper
remove_files ./ccl_offload_ex/imports/ccl_offload.v
add_files -norecurse ./hdl/ccl_offload.v
update_compile_order -fileset sources_1

# add elf file and associate it
add_files -norecurse $elf
update_compile_order -fileset sources_1
set_property SCOPED_TO_REF ccl_offload_bd [get_files -all -of_objects [get_fileset sources_1] $elf]
set_property SCOPED_TO_CELLS { control/microblaze_0 } [get_files -all -of_objects [get_fileset sources_1] $elf]

#run kernel packaging
reset_run synth_1
launch_runs synth_1 -jobs 12
wait_on_run [get_runs synth_1]
open_run synth_1 -name synth_1
rename_ref -prefix_all ccl_offload_
write_checkpoint ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
write_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc
close_design
package_project_dcp_and_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc ./ccl_offload_ex/ccl_offload Xilinx XCCL ccl_offload
package_xo  -xo_path ./ccl_offload_ex/exports/ccl_offload.xo -kernel_name ccl_offload -ip_directory ./ccl_offload_ex/ccl_offload -kernel_xml ./ccl_offload_ex/imports/kernel.xml

# close and exit
close_project
exit
