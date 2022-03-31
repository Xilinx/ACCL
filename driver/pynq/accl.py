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

from argparse import ArgumentError
import pynq
import math
import numpy as np
import warnings
import numpy as np
import ipaddress
from enum import IntEnum, unique
import zmq

class SimMMIO():
    def __init__(self, zmqsocket):
        self.base_addr = 0
        self.socket = zmqsocket

    # MMIO read request  {"type": 0, "addr": <uint>}
    # MMIO read response {"status": OK|ERR, "rdata": <uint>}
    def read(self, offset):
        self.socket.send_json({"type": 0, "addr": offset})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ MMIO read error"
        return ack["rdata"]

    # MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
    # MMIO write response {"status": OK|ERR}
    def write(self, offset, val):
        self.socket.send_json({"type": 1, "addr": offset, "wdata": val})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ MMIO write error"

class SimBuffer():
    next_free_address = 0
    def __init__(self, data, zmqsocket, physical_address=None):
        self.socket = zmqsocket
        self.data = data
        if physical_address is None:
            self.physical_address = SimBuffer.next_free_address
            # allocate on 4K boundaries
            # not sure how realistic this is, but it does help
            # work around some addressing limitations in RTLsim
            SimBuffer.next_free_address += math.ceil(data.nbytes/4096)*4096
        else:
            self.physical_address = physical_address
        self.device_address = self.physical_address
    
    # Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
    # Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
    def sync_from_device(self):
        self.socket.send_json({"type": 2, "addr": self.physical_address, "len": self.data.nbytes})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ mem buffer read error"
        self.data.view(np.uint8)[:] = ack["rdata"]

    # Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>}
    # Devicemem write response {"status": OK|ERR}
    def sync_to_device(self):
        self.socket.send_json({"type": 3, "addr": self.physical_address, "wdata": self.data.view(np.uint8).tolist()})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ mem buffer write error"

    def freebuffer(self):
        pass

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is not None:
                offset = self.data[:key.start].nbytes
            else:
                offset = 0
            return SimBuffer(self.data[key], self.socket, physical_address=self.physical_address+offset)
        else:
            return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

class SimDevice():
    def __init__(self, zmqadr="tcp://localhost:5555"):
        print("SimDevice connecting to ZMQ on", zmqadr)
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(zmqadr)
        self.mmio = SimMMIO(self.socket)
        self.devicemem = None
        self.rxbufmem = None
        self.networkmem = None
        print("SimDevice connected")

    # Call request  {"type": 4, arg names and values}
    # Call response {"status": OK|ERR}
    def call(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=[]):
        assert len(waitfor) == 0, "SimDevice does not support chaining"
        self.socket.send_json({ "type": 4,
                                "scenario": scenario,
                                "count": count,
                                "comm": comm,
                                "root_src_dst": root_src_dst,
                                "function": function,
                                "tag": tag,
                                "arithcfg": arithcfg,
                                "compression_flags": compression_flags,
                                "stream_flags": stream_flags,
                                "addr_0": addr_0,
                                "addr_1": addr_1,
                                "addr_2": addr_2})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ call error"

    def start(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=[]):
        assert len(waitfor) == 0, "SimDevice does not support chaining"
        self.socket.send_json({ "type": 4,
                                "scenario": scenario,
                                "count": count,
                                "comm": comm,
                                "root_src_dst": root_src_dst,
                                "function": function,
                                "tag": tag,
                                "arithcfg": arithcfg,
                                "compression_flags": compression_flags,
                                "stream_flags": stream_flags,
                                "addr_0": addr_0,
                                "addr_1": addr_1,
                                "addr_2": addr_2})
        return self

    def read(self, offset):
        return self.mmio.read(offset)

    def write(self, offset, val):
        return self.mmio.write(offset, val)

    def wait(self):
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ call error"

class AlveoDevice():
    def __init__(self, xclbin, board_idx=0, core_idx=0, mem=None):
        print(f"AlveoDevice connecting to board {board_idx} core {core_idx} xclbin {xclbin}")
        local_alveo = pynq.Device.devices[board_idx]
        self.ol = pynq.Overlay(xclbin, device=local_alveo)
        self.cclo = self.ol.__getattr__(f"ccl_offload_{core_idx}")
        self.hostctrl = self.ol.__getattr__(f"hostctrl_{core_idx}")
        self.mmio = self.cclo.mmio
        if mem is None:
            print("Best-effort attempt at identifying memories to use for RX buffers")
            if local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
                print("Detected U250 (xilinx_u250_gen3x16_xdma_shell_3_1)")
                self.devicemem   = self.ol.bank1
                self.rxbufmem    = [self.ol.bank0, self.ol.bank1, self.ol.bank2]
                self.networkmem  = self.ol.bank3
            elif local_alveo.name == 'xilinx_u250_xdma_201830_2':
                print("Detected U250 (xilinx_u250_xdma_201830_2)")
                self.devicemem   = self.ol.bank0
                self.rxbufmem    = self.ol.bank0
                self.networkmem  = self.ol.bank0
            elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
                print("Detected U280 (xilinx_u280_xdma_201920_3)")
                self.devicemem   = self.ol.HBM0
                self.rxbufmem    = [self.ol.HBM0, self.ol.HBM1, self.ol.HBM2, self.ol.HBM3, self.ol.HBM4, self.ol.HBM5] 
                self.networkmem  = self.ol.HBM6
        print("AlveoDevice connected")

    def read(self, offset):
        return self.mmio.read(offset)

    def write(self, offset, val):
        return self.mmio.write(offset, val)

    def call(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=[]):
        if self.hostctrl is not None:
            self.hostctrl.call(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=waitfor)
        else:
            raise Exception("Host calling not supported, no hostctrl found")
    def start(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=[]):
        if self.hostctrl is not None:
            return self.hostctrl.start(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=waitfor)
        else:
            raise Exception("Host calling not supported, no hostctrl found")

@unique
class CCLOp(IntEnum):
    config                  = 0
    copy                    = 1
    combine                 = 2
    send                    = 3
    recv                    = 4
    bcast                   = 5
    scatter                 = 6
    gather                  = 7
    reduce                  = 8
    allgather               = 9
    allreduce               = 10
    reduce_scatter          = 11
    ext_stream_krnl         = 12
    nop                     = 255

@unique
class CCLOCfgFunc(IntEnum):
    reset_periph         = 0
    enable_pkt           = 1
    set_timeout          = 2
    open_port            = 3
    open_con             = 4
    set_stack_type       = 5
    set_max_segment_size = 6

@unique
class ACCLReduceFunctions(IntEnum):
    SUM = 0

@unique
class ACCLCompressionFlags(IntEnum):
    NO_COMPRESSION = 0
    OP0_COMPRESSED = 1
    OP1_COMPRESSED = 2
    RES_COMPRESSED = 4
    ETH_COMPRESSED = 8

@unique
class ACCLStreamFlags(IntEnum):
    NO_STREAM = 0
    OP0_STREAM = 1
    RES_STREAM = 2

class ACCLArithConfig():
    def __init__(self, uncompressed_elem_bytes, compressed_elem_bytes, elem_ratio_log, 
                    compressor_tdest, decompressor_tdest, arith_is_compressed, arith_tdest):
        self.uncompressed_elem_bytes = uncompressed_elem_bytes
        self.compressed_elem_bytes = compressed_elem_bytes
        self.elem_ratio_log = elem_ratio_log
        self.compressor_tdest = compressor_tdest
        self.decompressor_tdest = decompressor_tdest
        self.arith_is_compressed = arith_is_compressed
        self.arith_nfunctions = len(arith_tdest)
        self.arith_tdest = arith_tdest

        #address where stored in exchange memory
        self.exchmem_addr = None

    @property
    def addr(self):
        assert self.exchmem_addr is not None
        return self.exchmem_addr

    def write(self, mmio, addr):
        self.exchmem_addr = addr
        mmio.write(addr, self.uncompressed_elem_bytes)
        addr += 4
        mmio.write(addr, self.compressed_elem_bytes)
        addr += 4
        mmio.write(addr, self.elem_ratio_log)
        addr += 4
        mmio.write(addr, self.compressor_tdest)
        addr += 4
        mmio.write(addr, self.decompressor_tdest)
        addr += 4
        mmio.write(addr, self.arith_nfunctions)
        addr += 4
        mmio.write(addr, self.arith_is_compressed)
        addr += 4
        for elem in self.arith_tdest:
            mmio.write(addr, elem)
            addr += 4
        return addr

ACCL_DEFAULT_ARITH_CONFIG = {
    ('float16', 'float16'): ACCLArithConfig(2, 2, 0, 0, 0, 0, [4]),
    ('float32', 'float16'): ACCLArithConfig(4, 2, 0, 0, 1, 1, [4]),
    ('float32', 'float32'): ACCLArithConfig(4, 4, 0, 0, 0, 0, [0]),
    ('float64', 'float64'): ACCLArithConfig(8, 8, 0, 0, 0, 0, [1]),
    ('int32'  , 'int32'  ): ACCLArithConfig(4, 4, 0, 0, 0, 0, [2]),
    ('int64'  , 'int64'  ): ACCLArithConfig(8, 8, 0, 0, 0, 0, [3]),
}

@unique
class ErrorCode(IntEnum):
    COLLECTIVE_OP_SUCCESS             = 0  
    DMA_MISMATCH_ERROR                = 1     
    DMA_INTERNAL_ERROR                = 2     
    DMA_DECODE_ERROR                  = 3  
    DMA_SLAVE_ERROR                   = 4 
    DMA_NOT_OKAY_ERROR                = 5     
    DMA_NOT_END_OF_PACKET_ERROR       = 6             
    DMA_NOT_EXPECTED_BTT_ERROR        = 7
    DMA_TIMEOUT_ERROR                 = 8             
    CONFIG_SWITCH_ERROR               = 9
    DEQUEUE_BUFFER_TIMEOUT_ERROR      = 10
    RECEIVE_TIMEOUT_ERROR             = 12
    DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR = 11
    DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH = 13
    DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR = 14
    COLLECTIVE_NOT_IMPLEMENTED        = 15
    RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID = 16
    OPEN_PORT_NOT_SUCCEEDED           = 17
    OPEN_COM_NOT_SUCCEEDED            = 18
    DMA_SIZE_ERROR                    = 19
    ARITH_ERROR                       = 20
    PACK_TIMEOUT_STS_ERROR            = 21
    PACK_SEQ_NUMBER_ERROR             = 22
    ARITHCFG_ERROR                    = 23
    KRNL_TIMEOUT_STS_ERROR            = 24
    KRNL_STS_COUNT_ERROR              = 25

TAG_ANY = 0xFFFF_FFFF
EXCHANGE_MEM_OFFSET_ADDRESS= 0x0
EXCHANGE_MEM_ADDRESS_RANGE = 0x2000
RETCODE_OFFSET = 0x1FFC
IDCODE_OFFSET = 0x1FF8
CFGRDY_OFFSET = 0x1FF4

class accl():
    """
    ACCL Python Driver
    """
    def __init__(self, ranks, local_rank, xclbin=None, protocol="TCP", board_idx=0, nbufs=16, bufsize=1024, mem=None, arith_config=ACCL_DEFAULT_ARITH_CONFIG, sim_sock=None, core_idx=0):
        assert xclbin is not None or sim_sock is not None, "Either simulation socket or xclbin must be provided"
        self.cclo = None
        #define supported types and corresponding arithmetic config
        self.arith_config = {}
        self.arithcfg_addr = 0
        #define an empty list of RX spare buffers
        self.rx_buffer_spares = []
        self.rx_buffer_size = 0
        self.rx_buffers_adr = EXCHANGE_MEM_OFFSET_ADDRESS
        #define buffers for POE
        self.tx_buf_network = None
        self.rx_buf_network = None
        #define another spare for general use (e.g. as accumulator for reduce/allreduce)
        self.utility_spare = None
        #define an empty list of communicators, to which users will add
        self.communicators = []
        self.communicators_addr = self.rx_buffers_adr
        self.check_return_value_flag = True
        #enable safety checks by default
        self.ignore_safety_checks = False
        #TODO: use description to gather info about where to allocate spare buffers
        self.segment_size = None
        #protocol being used
        self.protocol = protocol
        #flag to indicate whether we've finished config
        self.config_rdy = False

        # do initial config of alveo or connect to pipes if in sim mode
        self.sim_mode = False if sim_sock is None else True
        self.sim_sock = sim_sock
        if self.sim_mode:
            self.cclo = SimDevice(sim_sock)
        else:
            self.cclo = AlveoDevice(xclbin, board_idx=board_idx, core_idx=core_idx, mem=mem)

        print("CCLO HWID: {} at {}".format(hex(self.get_hwid()), hex(self.cclo.mmio.base_addr)))
        
        # check if the CCLO is configured
        assert self.cclo.read(CFGRDY_OFFSET) == 0, "CCLO appears configured, might be in use. Please reset the CCLO and retry"

        print("Configuring RX Buffers")
        self.setup_rx_buffers(nbufs, bufsize)
        print("Configuring a communicator")
        self.configure_communicator(ranks, local_rank)
        print("Configuring arithmetic")
        self.configure_arithmetic(configs=arith_config)

        # mark CCLO as configured (config memory written)
        self.cclo.write(CFGRDY_OFFSET, 1)
        self.config_rdy = True

        # set error timeout
        self.set_timeout(1_000_000)

        # start Ethernet infrastructure
        #Start (de)packetizer
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.enable_pkt)
        #set segmentation size equal to buffer size
        self.set_max_segment_size(bufsize)

        # set stack type
        if self.protocol == "UDP":
            self.use_udp()
        elif self.protocol == "TCP":
            if not self.sim_mode:
                self.tx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=self.cclo.networkmem)
                self.rx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=self.cclo.networkmem)
                self.tx_buf_network.sync_to_device()
                self.rx_buf_network.sync_to_device()
            self.use_tcp()
        elif self.protocol == "RDMA":
            raise ArgumentError("RDMA not supported yet")
        else:
            raise ArgumentError("Unrecognized Protocol")

        # start connections if using TCP
        if self.protocol == "TCP":
            print("Starting connections to communicator ranks")
            self.init_connection(comm_id=0)

        print("Accelerator ready!")


    class dummy_address:
        def __init__(self, adr=0):
            self.device_address = adr
            self.physical_address = adr
            self.dtype = None
            self.size = 0

    def dump_exchange_memory(self):
        print("exchange mem:")
        num_word_per_line=4
        for i in range(0,EXCHANGE_MEM_ADDRESS_RANGE, 4*num_word_per_line):
            memory = []
            for j in range(num_word_per_line):
                memory.append(hex(self.cclo.read(EXCHANGE_MEM_OFFSET_ADDRESS+i+(j*4))))
            print(hex(EXCHANGE_MEM_OFFSET_ADDRESS + i), memory)

    def deinit(self):
        print("Removing CCLO object at ",hex(self.cclo.mmio.base_addr))
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.reset_periph)

        for buf in self.rx_buffer_spares:
            buf.freebuffer()
        del self.rx_buffer_spares
        self.rx_buffer_spares = []

        if self.utility_spare is not None:
            self.utility_spare.freebuffer()
        del self.utility_spare
        self.utility_spare = None 

    #define CCLO arithmetic configurations
    def configure_arithmetic(self, configs=ACCL_DEFAULT_ARITH_CONFIG):
        assert len(self.communicators) > 0, "Communicators unconfigured, please call configure_communicator() first"
        addr = self.arithcfg_addr
        self.arith_config = configs
        for key in self.arith_config.keys():
            #write configuration into exchange memory
            addr = self.arith_config[key].write(self.cclo.mmio, addr)

    def setup_rx_buffers(self, nbufs, bufsize):
        addr = self.rx_buffers_adr
        self.rx_buffer_size = bufsize
        mem = self.cclo.rxbufmem
        if not isinstance(mem, list):
            mem = [mem] 
        for i in range(nbufs):
            # create, clear and sync buffers to device
            if not self.sim_mode:
                #try to cycle through different banks 
                buf = pynq.allocate((bufsize,), dtype=np.int8, target=mem[i % len(mem)])
                buf[:] = np.zeros((bufsize,), dtype=np.int8)
            else:
                buf = SimBuffer(np.zeros((bufsize,), dtype=np.int8), self.cclo.socket)
            buf.sync_to_device()

            self.rx_buffer_spares.append(buf)
            #program this buffer into the accelerator
            addr += 4
            self.cclo.write(addr, 0)
            addr += 4
            self.cclo.write(addr, self.rx_buffer_spares[-1].physical_address & 0xffffffff)
            addr += 4
            self.cclo.write(addr, (self.rx_buffer_spares[-1].physical_address>>32) & 0xffffffff)
            addr += 4
            self.cclo.write(addr, bufsize)
            # clear remaining fields
            for _ in range(4,8):
                addr += 4
                self.cclo.write(addr, 0)
        #NOTE: the buffer count HAS to be written last (offload checks for this) 
        self.cclo.write(self.rx_buffers_adr, nbufs)

        self.communicators_addr = addr+4
        if not self.sim_mode:
            self.utility_spare = pynq.allocate((bufsize,), dtype=np.int8, target=mem[0])
        else:
            self.utility_spare = SimBuffer(np.zeros((bufsize,), dtype=np.int8), self.cclo.socket)
    
    def dump_rx_buffers(self, nbufs=None):
        addr = self.rx_buffers_adr
        if nbufs is None:
            assert self.cclo.read(addr) == len(self.rx_buffer_spares)
            nbufs = len(self.rx_buffer_spares)
        print(f"CCLO address:{hex(self.cclo.mmio.base_addr)}")
        nbufs = min(len(self.rx_buffer_spares), nbufs)
        for i in range(nbufs):
            addr   += 4
            rstatus  = self.cclo.read(addr)
            addr   += 4
            addrl   =self.cclo.read(addr)
            addr   += 4
            addrh   = self.cclo.read(addr)
            addr   += 4
            maxsize = self.cclo.read(addr)
            #assert self.cclo.read(addr) == self.rx_buffer_size
            addr   += 4
            rxtag   = self.cclo.read(addr)
            addr   += 4
            rxlen   = self.cclo.read(addr)
            addr   += 4
            rxsrc   = self.cclo.read(addr)
            addr   += 4
            seq     = self.cclo.read(addr)
            
            if rstatus == 0 :
                status =  "NOT USED"
            elif rstatus == 1:
                status = "ENQUEUED"
            elif rstatus == 2:
                status = "RESERVED"
            else :
                status = "UNKNOWN"

            try:
                self.rx_buffer_spares[i].sync_from_device()
                if self.sim_mode:
                    content = str(self.rx_buffer_spares[i].buf.view(np.uint8))
                else:
                    content = str(self.rx_buffer_spares[i].view(np.uint8))
            except Exception :
                content= "xxread failedxx"
            buf_phys_addr = addrh*(2**32)+addrl
            print(f"SPARE RX BUFFER{i}:\t ADDR: {hex(buf_phys_addr)} \t STATUS: {status} \t OCCUPANCY: {rxlen}/{maxsize} \t  MPI TAG:{hex(rxtag)} \t SEQ: {seq} \t SRC:{rxsrc} \t DATA: {content}")

    def prepare_call(self, addr_0, addr_1, addr_2, compress_dtype=None):
        # no addresses, this is a config call
        # set dummy addresses where needed
        if addr_0 is None:
            addr_0 = self.dummy_address()
        if addr_1 is None:
            addr_1 = self.dummy_address()
        if addr_2 is None:
            addr_2 = self.dummy_address()
        # check data types of inputs and outputs to determine the arithmetic config and compression flags
        # if no explicit compression flag is set, conservatively perform transmission at the uncompressed
        # precision
        dtypes = {addr_0.dtype, addr_1.dtype, addr_2.dtype}
        dtypes.discard(None)
        if len(dtypes) == 0:
            #this must be a housekeeping call, no config needed
            arithcfg = 0
            compression_flags = ACCLCompressionFlags.NO_COMPRESSION
            return arithcfg, compression_flags, addr_0.device_address, addr_1.device_address, addr_2.device_address
        # if no compressed data type specified, set same as uncompressed
        compression_flags = ACCLCompressionFlags.NO_COMPRESSION
        if compress_dtype is None:
            # no ethernet compression
            if len(dtypes) == 1:
                # no operand compression
                single_dtype = dtypes.pop()
                arithcfg = self.arith_config[(single_dtype.name, single_dtype.name)]
            else:
                # with operand compression
                # determine compression dtype
                dt1 = dtypes.pop()
                dt2 = dtypes.pop()
                c_dt = dt1 if dt1.itemsize < dt2.itemsize else dt2
                u_dt = dt2 if dt1.itemsize < dt2.itemsize else dt1
                # determine which operand is compressed
                if addr_0.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.OP0_COMPRESSED
                if addr_1.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.OP1_COMPRESSED
                if addr_2.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.RES_COMPRESSED
                # set arithcfg
                arithcfg = self.arith_config[(u_dt.name, c_dt.name)]
        else:
            # we use ethernet compression
            compression_flags |= ACCLCompressionFlags.ETH_COMPRESSED
            if len(dtypes) == 1:
                # no operand compression
                arithcfg = self.arith_config[(dtypes.pop().name, compress_dtype.name)]
            else:
                assert compress_dtype in dtypes, "Unsupported data type combination"
                dtypes.discard(compress_dtype)
                # with operand compression
                c_dt = compress_dtype
                u_dt = dtypes.pop()
                # determine which operand is compressed
                if addr_0.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.OP0_COMPRESSED
                if addr_1.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.OP1_COMPRESSED
                if addr_2.dtype == c_dt:
                    compression_flags |= ACCLCompressionFlags.RES_COMPRESSED
                # set arithcfg
                arithcfg = self.arith_config[(u_dt.name, c_dt.name)]
        return arithcfg.addr, compression_flags, addr_0.device_address, addr_1.device_address, addr_2.device_address

    def call_async(self, scenario=CCLOp.nop, count=1, comm=0, root_src_dst=0, function=0, tag=TAG_ANY, compress_dtype=None, stream_flags=ACCLStreamFlags.NO_STREAM, addr_0=None, addr_1=None, addr_2=None, waitfor=[]):
        assert self.config_rdy, "CCLO not configured, cannot call"
        arithcfg, compression_flags, addr_0, addr_1, addr_2 = self.prepare_call(addr_0, addr_1, addr_2, compress_dtype)
        return self.cclo.start(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2, waitfor=waitfor)        

    def call_sync(self, scenario=CCLOp.nop, count=1, comm=0, root_src_dst=0, function=0, tag=TAG_ANY, compress_dtype=None, stream_flags=ACCLStreamFlags.NO_STREAM, addr_0=None, addr_1=None, addr_2=None):
        assert self.config_rdy, "CCLO not configured, cannot call"
        arithcfg, compression_flags, addr_0, addr_1, addr_2 = self.prepare_call(addr_0, addr_1, addr_2, compress_dtype)
        return self.cclo.call(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2)        

    def get_retcode(self):
        return self.cclo.read(RETCODE_OFFSET)

    def self_check_return_value(call):
        def wrapper(self, *args, **kwargs):
            handle = call(self, *args, **kwargs)
            if self.check_return_value_flag and handle is None: # if handle is none it means that the execution was synchronous
                self.check_return_value(call.__name__)
            else: #not possible to check return code if invoked async
                pass
            return handle
        return wrapper
    
    def check_return_value(self, label=""):
        retcode = self.get_retcode()
        if retcode != 0:
            try:
                error_msg = ErrorCode(retcode).name
            except:
                error_msg = f"UNKNOWN ERROR ({retcode})"
            raise Exception(f"CCLO @{hex(self.cclo.mmio.base_addr)}: during {label} {error_msg} you should consider resetting mpi_offload")

    def get_hwid(self):
        #TODO: add check
        return self.cclo.read(IDCODE_OFFSET) 

    def set_timeout(self, value, run_async=False, waitfor=[]):
        self.call_sync(scenario=CCLOp.config, count=value, function=CCLOCfgFunc.set_timeout)

    def init_connection(self, comm_id=0):
        print("Opening ports to communicator ranks")
        self.open_port(comm_id)
        print("Starting sessions to communicator ranks")
        self.open_con(comm_id)
    
    @self_check_return_value
    def open_port(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, comm=self.communicators[comm_id]["addr"], function=CCLOCfgFunc.open_port)
    
    @self_check_return_value
    def open_con(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, comm=self.communicators[comm_id]["addr"], function=CCLOCfgFunc.open_con)
    
    @self_check_return_value
    def use_udp(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_stack_type, count=0)
    
    @self_check_return_value
    def use_tcp(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_stack_type, count=1)   
    
    @self_check_return_value
    def set_max_segment_size(self, value=0):
        if value % 8 != 0:
            warnings.warn("ACCL: dma transaction must be divisible by 8 to use reduce collectives")
        elif value > self.rx_buffer_size:
            warnings.warn("ACCL: transaction size should be less or equal to configured buffer size!")
            return
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_max_segment_size, count=value)   
        self.segment_size = value

    @self_check_return_value
    def set_max_dma_in_flight(self, value=0):
     
        if value > 20:
            warnings.warn("ACCL: transaction size should be less or equal to configured buffer size!")
            return
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_max_dma_transactions, count=value)   

    def configure_communicator(self, ranks, local_rank):
        assert len(self.rx_buffer_spares) > 0, "RX buffers unconfigured, please call setup_rx_buffers() first"
        if len(self.communicators) == 0:
            addr = self.communicators_addr
        else:
            addr = self.communicators[-1]["addr"]
        communicator = {"local_rank": local_rank, "addr": addr, "ranks": ranks}
        self.cclo.write(addr,len(ranks))
        addr += 4
        self.cclo.write(addr,local_rank)
        for i in range(len(ranks)):
            addr += 4
            #ip string to int conversion from here:
            #https://stackoverflow.com/questions/5619685/conversion-from-ip-string-to-integer-and-backward-in-python
            self.cclo.write(addr, int(ipaddress.IPv4Address(ranks[i]["ip"])))
            addr += 4
            self.cclo.write(addr,ranks[i]["port"])
            #leave 2 32 bit space for inbound/outbound_seq_number
            addr += 4
            self.cclo.write(addr,0)
            addr +=4
            self.cclo.write(addr,0)
            addr += 4
            if "session_id" in ranks[i]:
                sess_id = ranks[i]["session_id"]
            else:
                sess_id = 0xFFFFFFFF
            self.cclo.write(addr, sess_id)
            addr += 4
            self.cclo.write(addr, ranks[i]["max_segment_size"])
        self.communicators.append(communicator)
        self.arithcfg_addr = addr + 4
        
    def dump_communicator(self):
        addr    = self.communicators_addr
        nr_ranks    = self.cclo.read(addr)
        addr +=4
        local_rank  = self.cclo.read(addr)
        print(f"Communicator. local_rank: {local_rank} \t number of ranks: {nr_ranks}.")
        for i in range(nr_ranks):
            addr +=4
            #ip string to int conversion from here:
            #https://stackoverflow.com/questions/5619685/conversion-from-ip-string-to-integer-and-backward-in-python
            ip_addr_rank = str(ipaddress.IPv4Address(self.cclo.read(addr)))
            addr += 4
            #when using the UDP stack, write the rank number into the port register
            #the actual port is programmed into the stack itself
            port = self.cclo.read(addr)
            #leave 2 32 bit space for inbound/outbound_seq_number
            addr += 4
            inbound_seq_number  = self.cclo.read(addr)
            addr +=4
            outbound_seq_number = self.cclo.read(addr)
            #a 32 bit integer is dedicated to session id 
            addr += 4
            session = self.cclo.read(addr)
            addr += 4
            max_seg_size = self.cclo.read(addr)
            print(f"> rank {i} (ip {ip_addr_rank}:{port} ; session {session} ; max segment size {max_seg_size}) : <- inbound_seq_number {inbound_seq_number}, -> outbound_seq_number {outbound_seq_number}")
   

    @self_check_return_value
    def nop(self, run_async=False, waitfor=[]):
        #calls the accelerator with no work. Useful for measuring call latency
        handle = self.call_async(scenario=CCLOp.nop, waitfor=waitfor)
        if run_async:
            return handle 
        else:
            handle.wait()

    @self_check_return_value
    def send(self, comm_id, srcbuf, count, dst, tag=TAG_ANY, from_fpga=False, stream_flags=ACCLStreamFlags.NO_STREAM, run_async=False, waitfor=[]):
        if not from_fpga:
            srcbuf.sync_to_device()
        handle = self.call_async(scenario=CCLOp.send, count=count, comm=self.communicators[comm_id]["addr"], root_src_dst=dst, tag=tag, stream_flags=stream_flags, addr_0=srcbuf, waitfor=waitfor)
        if run_async:
            return handle 
        else:
            handle.wait()
    
    @self_check_return_value
    def recv(self, comm_id, dstbuf, count, src, tag=TAG_ANY, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        handle = self.call_async(scenario=CCLOp.recv, count=count, comm=self.communicators[comm_id]["addr"], root_src_dst=src, tag=tag, addr_2=dstbuf, waitfor=waitfor)
        if run_async:
            return handle
        else:
            handle.wait()
        if not to_fpga:
            dstbuf.sync_from_device()

    @self_check_return_value
    def copy(self, srcbuf, dstbuf, count, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        # performs dstbuf = srcbuf
        if not from_fpga:
            srcbuf.sync_to_device()
        handle = self.call_async(scenario=CCLOp.copy, count=count, addr_0=srcbuf, addr_2=dstbuf, waitfor=waitfor)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            dstbuf.sync_from_device()

    @self_check_return_value
    def combine(self, count, func, val1, val2, result, val1_from_fpga=False, val2_from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        # TODO: check datatype support
        # performs acc = val + acc
        if not val1_from_fpga:
            val1.sync_to_device()
        if not val2_from_fpga:
            val2.sync_to_device()
        handle = self.call_async(scenario=CCLOp.combine, count=count, function=func, addr_0=val1, addr_1=val2, addr_2=result, waitfor=waitfor)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            result.sync_from_device()
    
    @self_check_return_value
    def external_stream_kernel(self, src_buf, dst_buf, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if src_buf.size <= 4:
            warnings.warn("size of buffer not compatible")
            return

        if not from_fpga:
            src_buf.sync_to_device()

        handle = self.call_async(scenario=CCLOp.ext_stream_krnl, count=src_buf.size, addr_0=src_buf, addr_1=dst_buf, waitfor=waitfor)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            dst_buf.sync_from_device()

    @self_check_return_value
    def bcast(self, comm_id, buf, count, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        comm = self.communicators[comm_id]
        is_root = comm["local_rank"] == root
        if not to_fpga and not(is_root) and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            warnings.warn("zero size buffer")
            return
        # sync the transmit source in one go
        if not from_fpga and is_root:
            buf.sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.bcast, count=count, comm=self.communicators[comm_id]["addr"], root_src_dst=root, addr_0=buf, waitfor=waitfor)]
        
        if run_async:
            return prevcall[0]
        
        prevcall[0].wait()
        if not to_fpga and not is_root:
            buf.sync_from_device()

    @self_check_return_value
    def scatter(self, comm_id, sbuf, rbuf, count, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            warnings.warn("zero size buffer")
            return
        comm        = self.communicators[comm_id]
        local_rank  = comm["local_rank"]
        p           = len(comm["ranks"])

        if not from_fpga and local_rank == root:
            sbuf[:count*p].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.scatter, count=count, comm=comm["addr"], root_src_dst=root, addr_0=sbuf, addr_2=rbuf[0:count], waitfor=waitfor)]

        if run_async:
            return prevcall[0]

        prevcall[0].wait()
        if not to_fpga:
            rbuf[0:count].sync_from_device()

    @self_check_return_value
    def gather(self, comm_id, sbuf, rbuf, count, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            warnings.warn("zero size buffer")
            return
        comm        = self.communicators[comm_id]
        local_rank  = comm["local_rank"]
        p           = len(comm["ranks"])

        if not self.ignore_safety_checks and (count + self.segment_size-1)//self.segment_size * p > len(self.rx_buffer_spares):
            warnings.warn("gather can't be executed safely with this number of spare buffers")
            return
        
        if not from_fpga:
            sbuf[0:count].sync_to_device()
            
        prevcall = [self.call_async(scenario=CCLOp.gather, count=count, comm=comm["addr"], root_src_dst=root, addr_0=sbuf, addr_2=rbuf, waitfor=waitfor)]
            
        if run_async:
            return prevcall[0]
        
        prevcall[0].wait()
        if not to_fpga and local_rank == root:
            rbuf[:count*p].sync_from_device()

    @self_check_return_value
    def allgather(self, comm_id, sbuf, rbuf, count, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            return
        comm    = self.communicators[comm_id]
        p       = len(comm["ranks"])

        if not self.ignore_safety_checks and (count + self.segment_size-1)//self.segment_size * p > len(self.rx_buffer_spares):
            warnings.warn("All gather can't be executed safely with this number of spare buffers")
            return
        
        if not from_fpga:
            sbuf[0:count].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.allgather, count=count, comm=comm["addr"], addr_0=sbuf, addr_2=rbuf, waitfor=waitfor)]

        if run_async:
            return prevcall[0]

        prevcall[0].wait()
        if not to_fpga:
            rbuf[:count*p].sync_from_device()

    #TODO: figure out if we need to mess with the datatypes
    # https://stackoverflow.com/questions/49135350/how-to-create-a-uint16-numpy-array-from-a-uint8-raw-image-data-array
    @self_check_return_value
    def reduce(self, comm_id, sbuf, rbuf, count, root, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            warnings.warn("zero size buffer")
            return

        comm        = self.communicators[comm_id]
        p           = len(comm["ranks"])
        local_rank  = comm["local_rank"]

        if not from_fpga:
            sbuf[0:count].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.reduce, count=count, comm=self.communicators[comm_id]["addr"], root_src_dst=root, function=func, addr_0=sbuf, addr_2=rbuf, waitfor=waitfor)]

        if run_async:
            return prevcall[0]
        
        prevcall[0].wait()
        if not to_fpga and local_rank == root:
            rbuf[0:count].sync_from_device()
 
    @self_check_return_value
    def allreduce(self, comm_id, sbuf, rbuf, count, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            return

        if not from_fpga:
            sbuf[0:count].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.allreduce, count=count, comm=self.communicators[comm_id]["addr"], function=func, addr_0=sbuf, addr_2=rbuf, waitfor=waitfor)]

        if run_async:
            return prevcall[0]

        prevcall[0].wait()
        if not to_fpga:
            rbuf[0:count].sync_from_device()
    
    @self_check_return_value
    def reduce_scatter(self, comm_id, sbuf, rbuf, count, func, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if count == 0:
            warnings.warn("zero size buffer")
            return

        comm        = self.communicators[comm_id]
        p           = len(comm["ranks"])
        local_rank  = comm["local_rank"]

        if not from_fpga:
            sbuf[0:count*p].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.reduce_scatter, count=count, comm=self.communicators[comm_id]["addr"], function=func, addr_0=sbuf, addr_2=rbuf, waitfor=waitfor)]

        if run_async:
            return prevcall[0]
        
        prevcall[0].wait()
        if not to_fpga:
            rbuf[0:count].sync_from_device()
