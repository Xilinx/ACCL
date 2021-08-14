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
# called: xsct tcl/generate_sw.tcl ccl_offload ccl_offload_ex/ccl_offload.hdf ./fw "Optimize most (-O3)"
set kernel_name    [lindex $argv 0]
set platform       [lindex $argv 1]
set repo_directory [lindex $argv 2]
set optimization_l [lindex $argv 3]
set domain_name mydomain
file mkdir vitis_ws
setws vitis_ws
repo -set $repo_directory
platform create -name $kernel_name -hw $platform
domain create -name $domain_name -proc control_microblaze_0
platform generate
app create -name ${kernel_name}_control -template {CCL Offload Control} -platform $kernel_name -proc control_microblaze_0
# reference: vitis XSCT doc https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/vpu1585821553007.html
#app config -name ${kernel_name}_control -info compiler-optimization
#possible values: "None (-O0)", "Optimize (-O1)", "Optimize more (-O2)", "Optimize most (-O3)", "Optimize for size (-Os)"
app config -name ${kernel_name}_control -set  compiler-optimization ${optimization_l}
app build  -name ${kernel_name}_control
exit
