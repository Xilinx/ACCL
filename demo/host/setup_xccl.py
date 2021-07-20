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

import os
import sys
import pynq
import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isfile, join, split
import socket
import struct

def configure_xccl(xclbin, board_idx):

    local_alveo = pynq.Device.devices[board_idx]
    ol=pynq.Overlay(xclbin, device=local_alveo)

    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = ol.DDR2
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem = ol.HBM0
    buf0 = pynq.allocate((1024*1024,), dtype=np.int8, target=devicemem)
    buf1 = pynq.allocate((1024*1024,), dtype=np.int8, target=devicemem)
    buf2 = pynq.allocate((1024*1024,), dtype=np.int8, target=devicemem)

    xccl_offload = ol.ccl_offload_inst
    print("Accelerator ready!")

    return ol, xccl_offload, buf0, buf1, buf2

def configure_rx_buffers_and_ranks(xccl_offload, nbufs, bufsize, ranks):

    #define a list of 16 buffers for RX, each 16k in size
    rx_list = []
    addr = 0x74
    xccl_offload.write(addr,nbufs)
    for i in range(nbufs):
        rx_list.append(pynq.allocate((bufsize,), dtype=np.int8, target=devicemem))
        #program this buffer into the accelerator
        addr += 4
        xccl_offload.write(addr, rx_list[-1].physical_address & 0xffffffff)
        addr += 4
        xccl_offload.write(addr, (rx_list[-1].physical_address>>32) & 0xffffffff)
        addr += 4
        xccl_offload.write(addr, bufsize)
    
    addr += 4
    xccl_offload.write(addr,len(ranks))
    for i in range(len(ranks)):
        addr += 4
        #ip string to int conversion from here:
        #https://stackoverflow.com/questions/5619685/conversion-from-ip-string-to-integer-and-backward-in-python
        xccl_offload.write(addr,struct.unpack("!I", socket.inet_aton(ranks[i]["ip"]))[0])
        addr += 4
        xccl_offload.write(addr,ranks[i]["port"])
    
    return rx_list
    
    