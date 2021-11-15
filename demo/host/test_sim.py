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

sys.path.append('../../driver/pynq/')
from accl import accl, ACCLReduceFunctions
from accl import SimBuffer
import argparse
import itertools

def get_buffers(count, op0_dt, op1_dt, res_dt, accl_inst):
    op0_buf = SimBuffer(np.zeros((count,), dtype=op0_dt), cclo_inst.cclo.socket)
    op1_buf = SimBuffer(np.zeros((count,), dtype=op1_dt), cclo_inst.cclo.socket)
    res_buf = SimBuffer(np.zeros((count,), dtype=res_dt), cclo_inst.cclo.socket)
    op0_buf.buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf.buf[:] = np.random.randn(count).astype(op1_dt)
    return op0_buf, op1_buf, res_buf

def test_copy(cclo_inst, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half, np.float64]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf.sync_to_device()
        res_buf.sync_to_device()
        cclo_inst.copy(op_buf, res_buf, count)
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Copy failed on pair ", op_dt, res_dt)
        else:
            print("Copy succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Copy succeeded")

def test_combine(cclo_inst, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op0_dt, op1_dt, res_dt in itertools.product(dt, repeat=3):
        op0_buf, op1_buf, res_buf = get_buffers(count, op0_dt, op1_dt, res_dt, cclo_inst)
        op0_buf.sync_to_device()
        op1_buf.sync_to_device()
        res_buf.sync_to_device()
        cclo_inst.combine(count, ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf)
        if not np.isclose(op0_buf.buf+op1_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Combine failed on pair ", op0_dt, op1_dt, res_dt)
        else:
            print("Combine succeeded on pair ", op0_dt, op1_dt, res_dt)
    if err_count == 0:
        print("Combine succeeded")

def test_sendrecv(cclo_inst, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf.sync_to_device()
        res_buf.sync_to_device()
        # send to self (effectively copy via external udp streams)
        cclo_inst.send(0, op_buf, count, 0, tag=5)
        cclo_inst.recv(0, res_buf, count, 0, tag=5)
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Send/recv failed on pair ", op_dt, res_dt)
        else:
            print("Send/recv succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Send/recv succeeded")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
    parser.add_argument('--nruns',   type=int, default=1,                help='How many times to run each test')
    parser.add_argument('--nbufs',   type=int, default=16,               help='number of spare buffers to configure each ccl_offload')
    parser.add_argument('--count',   type=int, default=1024,             help='How many B per buffer')
    parser.add_argument('--debug',   action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--all',     action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',     action='store_true', default=False, help='Run nop test')
    parser.add_argument('--combine', action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',    action='store_true', default=False, help='Run copy test')
    parser.add_argument('--sndrcv',  action='store_true', default=False, help='Run send/receive test')
    parser.add_argument('--nosync',  action='store_true', default=False, help='Run tests without syncing buffers')

    args = parser.parse_args()
    if args.all:
        args.nop     = True
        args.combine = True
        args.copy    = True

    #set a random seed to make it reproducible
    np.random.seed(2021)

    ranks = []
    for i in range(2):
        ranks.append({"ip": "127.0.0.1", "port": 17000})

    #configure FPGA and CCLO cores with the default 16 RX buffers of bsize KB each
    cclo_inst = accl(ranks, 0, protocol="UDP", sim_sock="tcp://localhost:5555")
    cclo_inst.dump_rx_buffers_spares()

    try:
        if args.combine:
            for i in range(args.nruns):
                test_combine(cclo_inst, args.count)

        if args.copy:
            for i in range(args.nruns):
                test_copy(cclo_inst, args.count)
        
        if args.sndrcv:
            for i in range(args.nruns):
                test_sendrecv(cclo_inst, args.count)

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        cclo_inst.dump_communicator()
        if not args.nosync:
            cclo_inst.dump_rx_buffers_spares()

    cclo_inst.deinit()
