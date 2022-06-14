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
from pyaccl.accl import accl, ACCLReduceFunctions, ACCLStreamFlags
import pynq
import argparse
import itertools
import math
from mpi4py import MPI

allocated_buffers = {}

def get_buffers(count, op0_dt, op1_dt, res_dt, accl_inst):
    try:
        op0_buf, op1_buf, res_buf = allocated_buffers[(op0_dt, op1_dt, res_dt)]
        if op0_buf.size >= count:
            op0_buf = op0_buf[0:count]
            op1_buf = op1_buf[0:count]
            res_buf = res_buf[0:count]
        else:
            op0_buf = pynq.allocate((count,), dtype=op0_dt, target=accl_inst.cclo.devicemem)
            op1_buf = pynq.allocate((count,), dtype=op1_dt, target=accl_inst.cclo.devicemem)
            res_buf = pynq.allocate((count,), dtype=res_dt, target=accl_inst.cclo.devicemem)
            allocated_buffers[(op0_dt, op1_dt, res_dt)] = (op0_buf, op1_buf, res_buf)
    except:
        op0_buf = pynq.allocate((count,), dtype=op0_dt, target=accl_inst.cclo.devicemem)
        op1_buf = pynq.allocate((count,), dtype=op1_dt, target=accl_inst.cclo.devicemem)
        res_buf = pynq.allocate((count,), dtype=res_dt, target=accl_inst.cclo.devicemem)
        allocated_buffers[(op0_dt, op1_dt, res_dt)] = (op0_buf, op1_buf, res_buf)

    op0_buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf[:] = np.random.randn(count).astype(op1_dt)
    res_buf[:] = np.random.randn(count).astype(res_dt)
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

def test_sendrecv(cclo_inst, world_size, local_rank, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        if len(dt) > 1 and op_dt == res_dt:
            continue
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        next_rank = (local_rank+1)%world_size
        prev_rank = (local_rank+world_size-1)%world_size
        print("Sending on ",local_rank," to ",next_rank)
        cclo_inst.send(op_buf, count, next_rank, tag=0)
        print("Receiving on ",local_rank," from ",prev_rank)
        cclo_inst.recv(res_buf, count, prev_rank, tag=0)
        print("Sending on ",local_rank," to ",prev_rank)
        cclo_inst.send(res_buf, count, prev_rank, tag=1)
        print("Receiving on ",local_rank," from ",next_rank)
        cclo_inst.recv(res_buf, count, next_rank, tag=1)
        if not np.isclose(op_buf.astype(res_dt), res_buf).all():
            err_count += 1
            print("Send/recv failed on pair ", op_dt, res_dt)
        else:
            print("Send/recv succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Send/recv succeeded")

def test_sendrecv_strm(cclo_inst, world_size, local_rank, count, dt = [np.float32]):
    #NOTE: this requires loopback on the external stream interface
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        next_rank = (local_rank+1)%world_size
        prev_rank = (local_rank+world_size-1)%world_size
        print("Sending from memory on ",local_rank," to stream on ",next_rank)
        cclo_inst.send(op_buf, count, next_rank, stream_flags=ACCLStreamFlags.RES_STREAM, tag=0)
        # recv is direct, no call required
        print("Sending from stream on ",local_rank," to memory on ",prev_rank)
        cclo_inst.send(res_buf, count, prev_rank, stream_flags=ACCLStreamFlags.OP0_STREAM, tag=5)
        print("Receiving in memory on ",local_rank," from stream on ",next_rank)
        cclo_inst.recv(res_buf, count, next_rank, tag=5)
        if not np.isclose(op_buf.astype(res_dt), res_buf).all():
            err_count += 1
            print("Send/recv failed on pair ", op_dt, res_dt)
        else:
            print("Send/recv succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Send/recv succeeded")

def test_sendrecv_fanin(cclo_inst, world_size, local_rank, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        if local_rank != 0:
            for i in range(len(op_buf)):
                op_buf[i] = i+local_rank
            print("Sending on ", local_rank, " to 0")
            cclo_inst.send(op_buf, count, 0, tag=0)
        else:
            for i in range(world_size):
                if i == local_rank:
                    continue
                print("Receiving on 0 from ", i)
                cclo_inst.recv(res_buf, count, i, tag=0)
                for j in range(len(op_buf)):
                    op_buf[j] = j+i
                if not np.isclose(op_buf, res_buf).all():
                    err_count += 1
                    print("Fan-in send/recv failed for sender rank", i, "on pair", op_dt, res_dt)
                else:
                    print("Fan-in send/recv succeeded for sender rank ", i, "on pair", op_dt, res_dt)
    if err_count == 0:
        print("Fan-in send/recv succeeded")

def test_bcast(cclo_inst, local_rank, root, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [42+i for i in range(len(op_buf))]
        cclo_inst.bcast(op_buf if root == local_rank else res_buf, count, root=root)

        if local_rank == root:
            print("Bcast succeeded on pair ", op_dt, res_dt)
        else:
            if not np.isclose(op_buf, res_buf).all():
                err_count += 1
                print("Bcast failed on pair ", op_dt, res_dt)
            else:
                print("Bcast succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Bcast succeeded")

def test_scatter(cclo_inst, world_size, local_rank, root, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count*world_size, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*i for i in range(op_buf.size)]
        cclo_inst.scatter(op_buf, res_buf, count, root=root)

        if not np.isclose(op_buf[local_rank*count:(local_rank+1)*count], res_buf[0:count]).all():
            err_count += 1
            print("Scatter failed on pair ", op_dt, res_dt)
        else:
            print("Scatter succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Scatter succeeded")

def test_gather(cclo_inst, world_size, local_rank, root, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count*world_size, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*(local_rank+i) for i in range(op_buf.size)]
        cclo_inst.gather(op_buf, res_buf, count, root=root)

        if local_rank == root:
            for i in range(world_size):
                if not np.isclose(res_buf[i*count:(i+1)*count], [1.0*(i+j) for j in range(count)]).all():
                    err_count += 1
                    print("Gather failed for src rank", i, "on pair ", op_dt, res_dt)
                else:
                    print("Gather succeeded for src rank", i, "on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Gather succeeded")

def test_allgather(cclo_inst, world_size, local_rank, count, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count*world_size, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*(local_rank+i) for i in range(op_buf.size)]
        cclo_inst.allgather(op_buf, res_buf, count)

        for i in range(world_size):
            if not np.isclose(res_buf[i*count:(i+1)*count], [1.0*(i+j) for j in range(count)]).all():
                err_count += 1
                print("Allgather failed for src rank", i, "on pair ", op_dt, res_dt)
            else:
                print("Allgather succeeded for src rank", i, "on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Allgather succeeded")

def test_reduce(cclo_inst, world_size, local_rank, root, count, func, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*i*(local_rank+1) for i in range(op_buf.size)]
        cclo_inst.reduce(op_buf, res_buf, count, root, func)

        if local_rank == root:
            if not np.isclose(res_buf, sum(range(world_size+1))*op_buf).all():
                err_count += 1
                print("Reduce failed on pair ", op_dt, res_dt)
            else:
                print("Reduce succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Reduce succeeded on pair ", op_dt, res_dt)

def test_reduce_scatter(cclo_inst, world_size, local_rank, count, func, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(world_size*count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*i for i in range(op_buf.size)]
        cclo_inst.reduce_scatter(op_buf, res_buf, count, func)

        full_reduce_result = world_size*op_buf
        if not np.isclose(res_buf[0:count], full_reduce_result[local_rank*count:(local_rank+1)*count]).all():
            err_count += 1
            print("Reduce-scatter failed on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Reduce-scatter succeeded on pair ", op_dt, res_dt)

def test_allreduce(cclo_inst, world_size, local_rank, count, func, dt = [np.float32]):
    err_count = 0
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*i for i in range(op_buf.size)]
        cclo_inst.allreduce(op_buf, res_buf, count, func)
        full_reduce_result = world_size*op_buf
        if not np.isclose(res_buf, full_reduce_result).all():
            err_count += 1
            print("Allreduce failed on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Allreduce succeeded on pair ", op_dt, res_dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests for ACCL (emulation mode)')
    parser.add_argument('--nruns',      type=int,            default=1,     help='How many times to run each test')
    parser.add_argument('--start_port', type=int,            default=5500,  help='Start of range of ports usable for sim')
    parser.add_argument('--count',      type=int,            default=16,    help='How many B per buffer')
    parser.add_argument('--rxbuf_size', type=int,            default=1,     help='How many KB per RX buffer')
    parser.add_argument('--board_idx',  type=int,            default=0,     help='Index of Alveo board, if multiple present')
    parser.add_argument('--xclbin',     type=str,            default="",    help='Path to xclbin, if present')
    parser.add_argument('--all',        action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',        action='store_true', default=False, help='Run nop test')
    parser.add_argument('--combine',    action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',       action='store_true', default=False, help='Run copy test')
    parser.add_argument('--sndrcv',     action='store_true', default=False, help='Run send/receive test')
    parser.add_argument('--sndrcv_strm', action='store_true', default=False, help='Run send/receive stream test')
    parser.add_argument('--sndrcv_fanin', action='store_true', default=False, help='Run send/receive fan-in test')
    parser.add_argument('--bcast',      action='store_true', default=False, help='Run bcast test')
    parser.add_argument('--scatter',    action='store_true', default=False, help='Run scatter test')
    parser.add_argument('--gather',     action='store_true', default=False, help='Run gather test')
    parser.add_argument('--allgather',     action='store_true', default=False, help='Run allgather test')
    parser.add_argument('--reduce',     action='store_true', default=False, help='Run reduce test')
    parser.add_argument('--reduce_scatter', action='store_true', default=False, help='Run reduce-scatter test')
    parser.add_argument('--allreduce',  action='store_true', default=False, help='Run all-reduce test')
    parser.add_argument('--reduce_func', type=int,           default=0,     help='Function index for reduce')
    parser.add_argument('--compression', action='store_true', default=False, help='Run test using compression')
    parser.add_argument('--fp16', action='store_true', default=False, help='Run test using fp16')
    parser.add_argument('--fp64', action='store_true', default=False, help='Run test using fp64')
    parser.add_argument('--int32', action='store_true', default=False, help='Run test using int32')
    parser.add_argument('--int64', action='store_true', default=False, help='Run test using int64')

    args = parser.parse_args()
    args.rxbuf_size = 1024*args.rxbuf_size #convert from KB to B
    if args.all:
        args.copy    = True
        args.combine = True
        args.sndrcv  = True
        args.sndrcv_strm = True
        args.sndrcv_fanin = True
        args.bcast   = True
        args.scatter = True
        args.gather = True
        args.allgather = True
        args.reduce = True
        args.reduce_scatter = True
        args.allreduce = True

    # get communicator size and our local rank in it
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()

    assert world_size == 3, "This test only applies to the triple-CCLO configuration on a single board"

    #set a random seed to make it reproducible
    np.random.seed(2021+local_rank)

    ranks = []
    for i in range(world_size):
        ranks.append({"ip": "127.0.0.1", "port": args.start_port+world_size+i, "session_id":i, "max_segment_size": args.rxbuf_size})

    #configure FPGA and CCLO cores with the default 16 RX buffers of size given by args.rxbuf_size
    print(f"AlveoDevice connecting to board {args.board_idx} core {local_rank} xclbin {args.xclbin}")
    local_alveo = pynq.Device.devices[args.board_idx]
    #this will program the FPGA if not already; when running under MPI, program the board ahead of time
    #with xbutil to avoid race conditions on the XCLBIN writes
    ol = pynq.Overlay(args.xclbin, device=local_alveo)
    #get handles to ACCL cores
    cclo_ip = ol.__getattr__(f"ccl_offload_{local_rank}")
    hostctrl_ip = ol.__getattr__(f"hostctrl_{local_rank}")
    #create a memory config corresponding to each CCLO
    #CCLOs are connected to HBM banks [0:5], [6:11], [12:17]
    # for simplicity we use the first bank in the range for user buffers,
    # second bank for RX buffers, and third bank for theoretical TCP buffers
    mems = [[ol.HBM0, ol.HBM1, ol.HBM2],
            [ol.HBM6, ol.HBM7, ol.HBM8],
            [ol.HBM12, ol.HBM13, ol.HBM14]]
    cclo_inst = accl(   ranks,
                        local_rank,
                        bufsize=args.rxbuf_size,
                        protocol="TCP",
                        overlay=ol,
                        cclo_ip=cclo_ip,
                        hostctrl_ip=hostctrl_ip,
                        mem = mems[local_rank]
                    )
    cclo_inst.set_timeout(10**8)
    #barrier here to make sure all the devices are configured before testing
    comm.barrier()

    types = [[np.float32]]
    if args.fp16:
        types = [[np.float16]]
    if args.fp64:
        types = [[np.float64]]
    if args.int32:
        types = [[np.int32]]
    if args.int64:
        types = [[np.int64]]
    if args.compression:
        types = [[np.float32, np.float16]]

    try:
        for i in range(args.nruns):
            if args.nop:
                cclo_inst.nop()
                comm.barrier()
            for dt in types:
                print("Testing dt ",dt)
                if args.combine:
                    test_combine(cclo_inst, args.count, dt=dt)
                    comm.barrier()
                if args.copy:
                    test_copy(cclo_inst, args.count, dt=dt)
                    comm.barrier()
                if args.sndrcv:
                    test_sendrecv(cclo_inst, world_size, local_rank, args.count, dt=dt)
                    comm.barrier()
                if args.sndrcv_strm:
                    test_sendrecv_strm(cclo_inst, world_size, local_rank, args.count, dt=dt)
                    comm.barrier()
                if args.sndrcv_fanin:
                    test_sendrecv_fanin(cclo_inst, world_size, local_rank, args.count, dt=dt)
                    comm.barrier()
                if args.bcast:
                    test_bcast(cclo_inst, local_rank, i, args.count, dt=dt)
                    comm.barrier()
                if args.scatter:
                    test_scatter(cclo_inst, world_size, local_rank, i, args.count, dt=dt)
                    comm.barrier()
                if args.gather:
                    test_gather(cclo_inst, world_size, local_rank, i, args.count, dt=dt)
                    comm.barrier()
                if args.allgather:
                    test_allgather(cclo_inst, world_size, local_rank, args.count, dt=dt)
                    comm.barrier()
                if args.reduce:
                    test_reduce(cclo_inst, world_size, local_rank, i, args.count, args.reduce_func, dt=dt)
                    comm.barrier()
                if args.reduce_scatter:
                    test_reduce_scatter(cclo_inst, world_size, local_rank, args.count, args.reduce_func, dt=dt)
                    comm.barrier()
                if args.allreduce:
                    test_allreduce(cclo_inst, world_size, local_rank, args.count, args.reduce_func, dt=dt)
                    comm.barrier()

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        cclo_inst.dump_rx_buffers()

    cclo_inst.deinit()
