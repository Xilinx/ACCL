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

# netStackType - UDP or TCP - type of POE attachment generated
# enableDMA - 0/1 - enables DMAs, providing support for send/recv from/to memory, and collectives
# enableArithmetic - 0/1 - enables arithmetic, providing support for reduction collectives and combine primitive
# enableCompression - 0/1 - enables compression feature
# enableExtKrnlStream - 0/1 - enables PL stream attachments, providing support for non-memory send/recv
# debugLevel - 0/1/2 - enables DEBUG/TRACE support for the control microblaze
set stacktype [lindex $::argv 1]
set en_dma [lindex $::argv 2]
set en_arith [lindex $::argv 3]
set en_compress [lindex $::argv 4]
set en_extkrnl [lindex $::argv 5]
set mb_debug_level [lindex $::argv 6]

# open project
open_project ./ccl_offload_ex/ccl_offload_ex.xpr
update_compile_order -fileset sim_1

set extra_sim_options ""
if { $en_arith == 1 } { set extra_sim_options "$extra_sim_options -d ARITH_ENABLE " }
if { $en_compress == 1 } { set extra_sim_options "$extra_sim_options -d COMPRESSION_ENABLE " }
if { $en_dma == 1 } { set extra_sim_options "$extra_sim_options -d DMA_ENABLE " }
if { $en_extkrnl == 1 } { set extra_sim_options "$extra_sim_options -d STREAM_ENABLE " }
if { $stacktype == "TCP" } { set extra_sim_options "$extra_sim_options -d TCP_ENABLE " }
if { $mb_debug_level > 0 } { set extra_sim_options "$extra_sim_options -d MB_DEBUG_ENABLE " }
set_property -name {xsim.compile.xvlog.more_options} -value $extra_sim_options -objects [get_filesets sim_1]
set_property -name {xsim.elaborate.xelab.more_options} -value {-dll} -objects [get_filesets sim_1]
set_property generate_scripts_only 1 [current_fileset -simset]
launch_simulation -scripts_only

# close and exit
close_project
exit