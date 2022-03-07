# /*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
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

import numpy as np
import sys
sys.path.append('../../driver/pynq/')
from accl import accl
from accl import SimBuffer
from mpi4py import MPI

def test_sendrecv(cclo_inst, world_size, local_rank, count):
    op_buf = SimBuffer(np.zeros((count,), dtype=np.float32), cclo_inst.cclo.socket)
    op_buf.sync_to_device()
    op_buf.buf[:] = np.arange(count).astype(np.float32)
    res_buf = SimBuffer(np.zeros((count,), dtype=np.float32), cclo_inst.cclo.socket)
    res_buf.sync_to_device()
    if local_rank == 0:
        print("Start send")
        cclo_inst.send(0, op_buf, count, 1, tag=0)
        print("Finish send")
    else:
        print("Start recv")
        cclo_inst.recv(0, res_buf, count, 0, tag=0)
        print("Finish recv")
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            print("test failed")
        else:
            print("test successful")
    print("Wait for other processes...")
    comm.barrier()

if __name__ == "__main__":
    # get communicator size and our local rank in it
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()

    if world_size != 2:
        print("Run with two processes!")

    #set a random seed to make it reproducible
    np.random.seed(2021+local_rank)

    ranks = []
    for i in range(world_size):
        ranks.append({"ip": "127.0.0.1", "port": 5500 + world_size + i, "session_id":i, "max_segment_size": 1})

    #configure FPGA and CCLO cores with the default 16 RX buffers of size given by args.rxbuf_size
    cclo_inst = accl(ranks, local_rank, bufsize=1, protocol="TCP", sim_sock="tcp://localhost:"+str(5500 + local_rank))
    cclo_inst.set_timeout(10**8)
    print("finish setup")
    #barrier here to make sure all the devices are configured before testing
    comm.barrier()
    print("start test")
    test_sendrecv(cclo_inst, world_size, local_rank, 16)

    cclo_inst.deinit()
