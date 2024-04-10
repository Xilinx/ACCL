#!/bin/bash
# /*******************************************************************************
#  Copyright (C) 2024 Advanced Micro Devices, Inc
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

MODES=(
    axis3x
    udp
    tcp
    coyote_tcp
    coyote_rdma
)

PLATFORMS=(
    xilinx_u55c_gen3x16_xdma_3_202210_1
    xilinx_u50_gen3x16_xdma_5_202210_1
    xilinx_u200_gen3x16_xdma_2_202110_1
    xilinx_u280_gen3x16_xdma_1_202211_1
    xilinx_u250_gen3x16_xdma_4_1_202210_1
)

# build for any combination of mode and platform
for mode in ${MODES[@]}; do
    for platform in ${PLATFORMS[@]}; do
        make -j$(nproc) MODE=$mode PLATFORM=$platform > build_${mode}_${platform}.log
    done
done
