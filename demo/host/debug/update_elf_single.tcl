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
set target [lindex $argv 0]
connect -xvc 127.0.0.1
after 3000
targets
targets $target
rst -proc
dow ../../../kernel/vitis_ws/ccl_offload_control/Debug/ccl_offload_control.elf
con
puts -nonewline "Waiting... hit enter to continue"
flush stdout
gets stdin