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

proc swapp_get_name {} {
  return "CCL Offload Control"
}

proc swapp_get_description {} {
  return "CCL Offload control application."
}

proc swapp_is_supported_hw {} {
  return 1
}

proc swapp_is_supported_sw {} {
  set os [hsi::get_os]
  if [string equal $os "standalone"] {
    return 1
  }
  return 0
}

proc swapp_generate {} {
}

proc swapp_get_linker_constraints {} {
  return ""
}
