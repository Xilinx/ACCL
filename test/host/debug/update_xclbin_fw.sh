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

#Arguments: <old xclbin> <vivado project> <new elf> <new xclbin>

#extract bitstream from xclbin on first argument
xclbinutil --force --dump-section BITSTREAM:RAW:bitstream.bit --input $1

#use Vivado to extract descriptor.mmi file from the project
echo "set mmi [lindex $::argv 0]" > extract_mmi.tcl
echo "open_run impl_1" >> extract_mmi.tcl
echo "write_mem_info \$mmi" >> extract_mmi.tcl
vivado $2 -mode batch -source extract_mmi.tcl -tclargs descriptor.mmi

#update the new executable in the bitstream
updatemem -force --meminfo descriptor.mmi --data $3 --bit bitstream.bit --proc pfm_top_i/dynamic_region/ccl_offload_0/ccl_offload_bd_i/control/microblaze_0 --out bitstream_updated.bit

#write bitstream to xclbin
xclbinutil --force --replace-section BITSTREAM:RAW:bitstream_updated.bit --input file.xclbin --o $4

#remove temp files
rm bitstream.bit extract_mmi.tcl descriptor.mmi bitstream_updated.bit