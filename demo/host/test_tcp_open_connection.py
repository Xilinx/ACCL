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
from cclo import *
import json 
import argparse
import random
from queue import Queue
import threading
import time

def configure_accl(xclbin, board_idx, nbufs=16, bufsize=1024*1024):
    rank_id = 0
    size = 1

    local_alveo = pynq.Device.devices[board_idx]
    print("local_alveo: {}".format(local_alveo.name))
    ol=pynq.Overlay(xclbin, device=local_alveo)

    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
        devicemem = [ol.bank1]
        rxbufmem = [ol.bank1]
        networkmem = ol.bank1
    elif local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = [ol.bank0]
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem = [ol.HBM0]
        rxbufmem = [ ol.HBM1, ol.HBM2, ol.HBM3, ol.HBM4, ol.HBM5 ]
        networkmem = ol.HBM6

    cclo           = ol.ccl_offload_0
    network_kernel = ol.network_krnl_0

    print("CCLO {} HWID: {} at {}".format(rank_id, hex(cclo.get_hwid()), hex(cclo.mmio.base_addr)))
    
    global ranks
    ip_network_str = []
    ip_network = []
    port = []
    ranks = []

    arp_addr = []
    for i in range (size):
        ip_network_str.append("10.1.212.{}".format(151+i))
        port.append(5001+i)
        ranks.append({"ip": ip_network_str[i], "port": port[i]})
        ip_network.append(int(ipaddress.IPv4Address(ip_network_str[i])))

        arp_addr.append(ip_network[i])
        
    cclo.use_tcp()
    print(f"CCLO {rank_id}: Configuring RX Buffers")
    cclo.setup_rx_buffers(nbufs, bufsize, rxbufmem)
    print(f"CCLO {rank_id}: Configuring a communicator")
    cclo.configure_communicator(ranks, rank_id)
    print(f"CCLO {rank_id}: Configuring network stack")
   
    #assign 64 MB network tx and rx buffer
    tx_buf_network = pynq.allocate((128*1024*1024,), dtype=np.int8, target=networkmem)
    rx_buf_network = pynq.allocate((128*1024*1024,), dtype=np.int8, target=networkmem)
    
    tx_buf_network.sync_to_device()
    rx_buf_network.sync_to_device()
    
    start = time.perf_counter()
    print(f"CCLO {rank_id}: Launch network kernel, ip {hex(ip_network[rank_id])}, board number {rank_id}, arp {hex(arp_addr[rank_id])}")
    ret = network_kernel.start_sw(ip_network[rank_id], rank_id, arp_addr[rank_id], tx_buf_network, rx_buf_network)
    end = time.perf_counter()
    print(f"{end-start}: returned from network kernel {ret}")
    ret.wait()
    end = time.perf_counter()
    print(f"{end-start}: wait complete from network kernel {ret}")
    
    #to synchronize the processes
    #comm.barrier()
 
    # pdb.set_trace()
    print(f"CCLO {rank_id}: open port")
    cclo.open_port(0)


    return ol, cclo, devicemem, tx_buf_network, rx_buf_network


def deinit_system():
    global cclo_inst
    cclo_inst.deinit()

def reinit():
    global cclo_inst
    cclo_inst.deinit()
    ol, cclo_inst, devicemem = configure_accl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=max(16*1024, args.bsize))

def allocate_buffers(n, bsize, devicemem):
    tx_buf = []
    rx_buf = []
    
    tx_buf.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[0]))
    rx_buf.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[0]))

    for i, buf in enumerate(rx_buf):
        print(f'rx_buf {i}',hex(buf.device_address))
    for i, buf in enumerate(tx_buf):
        print(f'tx_buf {i}',hex(buf.device_address))

    return tx_buf, rx_buf

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
    parser.add_argument('--xclbin',         type=str, default=None,             help='Accelerator image file (xclbin)', required=True)
    parser.add_argument('--device_index',   type=int, default=1,                help='Card index')
    parser.add_argument('--nruns',          type=int, default=30,               help='How many times to run each test')
    parser.add_argument('--nbufs',          type=int, default=16,               help='number of spare buffers to configure each ccl_offload')
    parser.add_argument('--naccel',         type=int, default=4,                help='number of ccl_offload to test ')
    parser.add_argument('--bsize',          type=int, default=1024,             nargs="+" ,    help='How many KB per buffer')
    parser.add_argument('--segment_size',   type=int, default=1024,             nargs="+" ,    help='How many KB per spare_buffer')
    args = parser.parse_args()  

    try:
        
        #configure FPGA and CCLO cores with the default 16 RX buffers of bsize KB each
        ol, cclo_inst, devicemem, tx_buf_network, rx_buf_network = configure_accl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=max(1024, args.bsize))

        tx_buf, rx_buf = allocate_buffers(args.naccel, args.bsize, devicemem)

        cclo_inst.set_timeout(10000)
        cclo_inst.nop()
        #input("press any key to start")
        #set a random seed to make it reproducible
        np.random.seed(2021)
        global ranks
        print(ranks)
        start = time.perf_counter()
        for _ in range(args.nruns):
            for a_dictionary in ranks:
                
                import socket
                a_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                a_sock.connect((a_dictionary["ip"],a_dictionary["port"]))
                end = time.perf_counter()
                print( end-start, "[s]"," connected from", a_sock.getsockname(),", test passed.")
                del a_sock
            time.sleep(5)


        
    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)

    deinit_system()