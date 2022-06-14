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
import numpy as np
import time
sys.path.append('../../..//driver/pynq/src/pyaccl/')
from accl import accl, ACCLReduceFunctions, ACCLStreamFlags
sys.path.append('../../hardware/xup_vitis_network_example/Notebooks/')
import pynq
import argparse
import itertools
import math

def get_buffers(count, op0_dt, op1_dt, res_dt, accl_inst):
    op0_buf = pynq.allocate((count,), dtype=op0_dt, target=accl_inst.cclo.devicemem)
    op1_buf = pynq.allocate((count,), dtype=op1_dt, target=accl_inst.cclo.devicemem)
    res_buf = pynq.allocate((count,), dtype=res_dt, target=accl_inst.cclo.devicemem)
    op0_buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf[:] = np.random.randn(count).astype(op1_dt)
    return op0_buf, op1_buf, res_buf

def test_copy(cclo_inst, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        cclo_inst.copy(op_buf, res_buf, count)
        if not np.isclose(op_buf.astype(res_dt), res_buf, atol=1e-02).all():
            err_count += 1
            print("Copy failed on pair ", op_dt, res_dt)
        else:
            print("Copy succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Copy succeeded")

def test_combine(cclo_inst, count, dt = [np.float32]):
    err_count = 0
    for op0_dt, op1_dt, res_dt in itertools.product(dt, repeat=3):
        op0_buf, op1_buf, res_buf = get_buffers(count, op0_dt, op1_dt, res_dt, cclo_inst)
        cclo_inst.combine(count, ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf)
        if not np.isclose(op0_buf+op1_buf, res_buf, atol=1e-02).all():
            err_count += 1
            print("Combine failed on pair ", op0_dt, op1_dt, res_dt)
        else:
            print("Combine succeeded on pair ", op0_dt, op1_dt, res_dt)
    if err_count == 0:
        print("Combine succeeded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests for ACCL initialization')
    parser.add_argument('--port', type=int,            default=5500,  help='Start of range of ports usable for sim')
    parser.add_argument('--count',      type=int,            default=16,    help='How many B per buffer')
    parser.add_argument('--rxbuf_size', type=int,            default=1,     help='How many KB per RX buffer')
    parser.add_argument('--board_idx',  type=int,            default=0,     help='Index of Alveo board, if multiple present')
    parser.add_argument('--core_idx',   type=int,            default=0,     help='Index of CCLO CU, if multiple present')
    parser.add_argument('--xclbin',     type=str,            default="",    help='Path to xclbin, if present')
    parser.add_argument('--all',        action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',        action='store_true', default=False, help='Run nop test')
    parser.add_argument('--combine',    action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',       action='store_true', default=False, help='Run copy test')
    parser.add_argument('--fp16', action='store_true', default=False, help='Run test using fp16')
    parser.add_argument('--fp64', action='store_true', default=False, help='Run test using fp64')
    parser.add_argument('--int32', action='store_true', default=False, help='Run test using int32')
    parser.add_argument('--int64', action='store_true', default=False, help='Run test using int64')

    args = parser.parse_args()
    args.rxbuf_size = 1024*args.rxbuf_size #convert from KB to B
    if args.all:
        args.copy    = True
        args.combine = True

    #set a random seed to make it reproducible
    np.random.seed(2021)

    ranks = [{"ip": "192.168.0.1", "port": args.port, "session_id":0, "max_segment_size": args.rxbuf_size},
             {"ip": "192.168.0.2", "port": args.port, "session_id":1, "max_segment_size": args.rxbuf_size}]

    #configure FPGA and CCLO cores with the default 16 RX buffers of size given by args.rxbuf_size
    print(f"AlveoDevice connecting to board {args.board_idx} core {args.core_idx} xclbin {args.xclbin}")
    local_alveo = pynq.Device.devices[args.board_idx]
    #this will program the FPGA if not already; when running under MPI, program the board ahead of time
    #with xbutil to avoid race conditions on the XCLBIN writes
    ol = pynq.Overlay(args.xclbin, device=local_alveo)

    #get handles to ACCL cores
    cclo_ip = ol.__getattr__(f"ccl_offload_{args.core_idx}")
    hostctrl_ip = ol.__getattr__(f"hostctrl_{args.core_idx}")
    #create a memory config corresponding to each CCLO
    #CCLO is connected to DDR banks [0:2]
    # for simplicity we use the first bank for everything
    cclo_inst = accl(   ranks,
                        0,
                        bufsize=args.rxbuf_size,
                        protocol="UDP",
                        overlay=ol,
                        cclo_ip=cclo_ip,
                        hostctrl_ip=hostctrl_ip,
                        mem = [ol.HBM0, ol.HBM0, ol.HBM0]
                    )
    cclo_inst.set_timeout(10**8)

    types = [[np.float32]]
    if args.fp16:
        types = [[np.float16]]
    if args.fp64:
        types = [[np.float64]]
    if args.int32:
        types = [[np.int32]]
    if args.int64:
        types = [[np.int64]]

    try:
        if args.nop:
            cclo_inst.nop()
        for dt in types:
            print("Testing dt ",dt)
            if args.combine:
                test_combine(cclo_inst, args.count, dt=dt)
            if args.copy:
                test_copy(cclo_inst, args.count, dt=dt)

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        cclo_inst.dump_rx_buffers()

    cclo_inst.deinit()
