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

# add elf file and associate it
add_files -fileset sources_1 -norecurse $elf
add_files -fileset sim_1 -norecurse $elf
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
set_property SCOPED_TO_REF ccl_offload_bd [get_files -all -of_objects [get_fileset sources_1] $elf]
set_property SCOPED_TO_CELLS { cclo/control/microblaze_0 } [get_files -all -of_objects [get_fileset sources_1] $elf]
set_property SCOPED_TO_REF ccl_offload_bd [get_files -all -of_objects [get_fileset sim_1] $elf]
set_property SCOPED_TO_CELLS { cclo/control/microblaze_0 } [get_files -all -of_objects [get_fileset sim_1] $elf]

# close and exit
close_project
exit
