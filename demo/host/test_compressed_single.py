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

import sys
import warnings
import numpy as np
sys.path.append('../../driver/pynq/')
from cclo import *
import json 
import argparse
import random
import time
import itertools

def configure_xccl(xclbin, board_idx, nbufs=16, bufsize=1024):  
    local_alveo = pynq.Device.devices[board_idx]
    ol=pynq.Overlay(xclbin, device=local_alveo)

    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = ol.__getattr__(f"bank0")

    cclo = ol.__getattr__(f"ccl_offload_0")
    print("CCLO HWID: {} at {}".format(hex(cclo.get_hwid()), hex(cclo.mmio.base_addr)))

    ranks = [{"ip": "127.0.0.1", "port": i} for i in range(2)]
    print(devicemem)
    
    cclo.use_udp()
    print("Configuring RX Buffers")
    cclo.setup_rx_buffers(nbufs, bufsize, devicemem)
    print("Configuring a communicator")
    cclo.configure_communicator(ranks, 0)
    print("Configuring arithmetic")
    cclo.configure_arithmetic()

    # set error timeout
    cclo.set_timeout(1_000_000)

    print("Accelerator ready!")

    return ol, cclo, devicemem

def get_buffers(count, op0_dt, op1_dt, res_dt, devicemem):
    op0_buf = pynq.allocate((count,), dtype=op0_dt, target=devicemem)
    op1_buf = pynq.allocate((count,), dtype=op1_dt, target=devicemem)
    res_buf = pynq.allocate((count,), dtype=res_dt, target=devicemem)
    op0_buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf[:] = np.random.randn(count).astype(op1_dt)
    return op0_buf, op1_buf, res_buf

def test_copy(cclo_inst, devicemem):
    err_count = 0
    dt = [np.float32, np.half, np.float64]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(1024, op_dt, op_dt, res_dt, devicemem)
        cclo_inst.copy(op_buf, res_buf)
        if not np.isclose(op_buf, res_buf).all():
            err_count += 1
            print("Copy failed on pair ",dt_pair)
    if err_count == 0:
        print("Copy succeeded")

def test_combine(cclo_inst, devicemem):
    for op0_dt, op1_dt, res_dt in itertools.product([np.float32, np.half], repeat=3):
        op0_buf, op1_buf, res_buf = get_buffers(1024, op0_dt, op1_dt, res_dt, devicemem)
        sum_fp = op0_buf.astype(np.float32) + op1_buf.astype(np.float32)

        cclo_inst.combine(ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf)

        assert np.allclose(res_buf.astype(np.float32), sum_fp)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
    parser.add_argument('--xclbin',         type=str, default=None,             help='Accelerator image file (xclbin)', required=True)
    parser.add_argument('--device_index',   type=int, default=0,                help='Card index')
    parser.add_argument('--nruns',          type=int, default=1,                help='How many times to run each test')
    parser.add_argument('--nbufs',          type=int, default=16,               help='number of spare buffers to configure each ccl_offload')
    parser.add_argument('--bsize',          type=int, default=1024,             help='How many B per user buffer')
    parser.add_argument('--segment_size',   type=int, default=1024,             help='How many B per spare buffer')
    parser.add_argument('--single_bank',    action='store_true', default=False, help='use a single memory bank per CCL_Offload instance')
    parser.add_argument('--debug',          action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--all',            action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',            action='store_true', default=False, help='Run nop test')
    parser.add_argument('--accumulate',     action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',           action='store_true', default=False, help='Run copy test')

    args = parser.parse_args()
    if args.all:
        args.nop        = True
        args.accumulate = True
        args.copy       = True
        
    #configure FPGA and CCLO cores with the default 16 RX buffers of bsize KB each
    ol, cclo_inst, devicemem = configure_xccl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=args.segment_size)
    cclo_inst.dump_rx_buffers_spares()

    try:
        #set a random seed to make it reproducible
        np.random.seed(2021)

        if args.accumulate:
            for i in range(args.nruns):
                test_acc(cclo_inst, devicemem)

        if args.copy:
            for i in range(args.nruns):
                test_copy(cclo_inst, devicemem)   

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        cclo_inst.dump_communicator()
        cclo_inst.dump_rx_buffers_spares()

    cclo_inst.deinit()
