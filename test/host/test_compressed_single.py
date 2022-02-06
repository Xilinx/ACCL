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
from accl import accl
import argparse
import itertools

def get_buffers(count, op0_dt, op1_dt, res_dt, devicemem):
    op0_buf = pynq.allocate((count,), dtype=op0_dt, target=devicemem)
    op1_buf = pynq.allocate((count,), dtype=op1_dt, target=devicemem)
    res_buf = pynq.allocate((count,), dtype=res_dt, target=devicemem)
    op0_buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf[:] = np.random.randn(count).astype(op1_dt)
    return op0_buf, op1_buf, res_buf

def test_copy(cclo_inst, remote=False):
    err_count = 0
    dt = [np.float32, np.half, np.float64]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(1024, op_dt, op_dt, res_dt, cclo_inst.devicemem)
        cclo_inst.copy(op_buf, res_buf, from_fpga=remote, to_fpga=remote)
        if not remote:
            if not np.isclose(op_buf, res_buf).all():
                err_count += 1
                print("Copy failed on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Copy succeeded")

def test_combine(cclo_inst, remote=False):
    for op0_dt, op1_dt, res_dt in itertools.product([np.float32, np.half], repeat=3):
        op0_buf, op1_buf, res_buf = get_buffers(1024, op0_dt, op1_dt, res_dt, cclo_inst.devicemem)
        sum_fp = op0_buf.astype(np.float32) + op1_buf.astype(np.float32)

        cclo_inst.combine(ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf, val1_from_fpga=remote, val2_from_fpga=remote, to_fpga=remote)

        if not remote:
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
    parser.add_argument('--combine',        action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',           action='store_true', default=False, help='Run copy test')
    parser.add_argument('--nosync',         action='store_true', default=False, help='Run tests without syncing buffers')

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
    cclo_inst = accl(args.xclbin, ranks, 0, protocol="UDP", board_idx=args.device_index)

    try:
        if not args.nosync:
            cclo_inst.dump_rx_buffers_spares()

        if args.combine:
            for i in range(args.nruns):
                test_combine(cclo_inst, remote=args.nosync)

        if args.copy:
            for i in range(args.nruns):
                test_copy(cclo_inst, remote=args.nosync)

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
