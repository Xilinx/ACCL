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

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr

# replace the default top with the modified wrapper
remove_files ./ccl_offload_ex/imports/ccl_offload.v
add_files -norecurse ./hdl/ccl_offload.v -force
update_compile_order -fileset sources_1

# add elf file and associate it
remove_files  ./ccl_offload_ex/ccl_offload_ex.sdk/ccl_offload_control/Debug/ccl_offload_control.elf
add_files -norecurse $elf -force
set_property SCOPED_TO_REF ccl_offload_bd [get_files -all -of_objects [get_fileset sources_1] $elf]
set_property SCOPED_TO_CELLS { control/microblaze_0 } [get_files -all -of_objects [get_fileset sources_1] $elf]

#run kernel packaging
source -notrace ./ccl_offload_ex/imports/package_kernel.tcl
reset_run synth_1
launch_runs synth_1 -jobs 12
wait_on_run [get_runs synth_1]
open_run synth_1 -name synth_1
rename_ref -prefix_all ccl_offload_
write_checkpoint ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp
write_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc
close_design
package_project_dcp_and_xdc ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.dcp ./ccl_offload_ex/ccl_offload_ex.runs/synth_1/packaged.xdc ./ccl_offload_ex/ccl_offload Xilinx ACCL ccl_offload
package_xo  -xo_path ./ccl_offload_ex/exports/ccl_offload.xo -kernel_name ccl_offload -ip_directory ./ccl_offload_ex/ccl_offload -kernel_xml ./xml/kernel.xml

# close and exit
close_project
exit
