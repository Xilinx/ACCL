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

import pynq
from pynq import DefaultIP
import os
import sys
import numpy as np
import struct
import warnings
import numpy as np
import ipaddress
from enum import IntEnum, unique

@unique
class CCLOp(IntEnum):
    config                  = 0
    copy                    = 1
    sum                     = 2
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
    enable_irq               = 0
    disable_irq              = 1
    reset_periph             = 2
    enable_pkt               = 3
    set_timeout              = 4
    init_connection          = 5
    open_port                = 6
    open_con                 = 7
    use_tcp_stack            = 8
    use_udp_stack            = 9
    start_profiling          = 10
    end_profiling            = 11
    set_dma_transaction_size = 12
    set_max_dma_transactions = 13

@unique
class ACCLReduceFunctions(IntEnum):
    SUM = 0

class ACCLArithConfig():
    def __init__(self, s2t_tdest, src_op_bits, t2d_tdest, dst_op_bits, arith_op_tdest, arith_op_bits):
        self.s2t_tdest = s2t_tdest
        self.src_op_bits = src_op_bits
        self.t2d_tdest = t2d_tdest
        self.dst_op_bits = dst_op_bits
        self.arith_op_tdest = arith_op_tdest
        self.arith_op_bits = arith_op_bits
        #address where stored in exchange memory
        self.exchmem_addr = None

    @property
    def addr(self):
        assert self.exchmem_addr is not None
        return self.exchmem_addr

    def write(self, mmio, addr):
        self.exchmem_addr = addr
        mmio.write(addr, self.s2t_tdest)
        addr += 4
        mmio.write(addr, self.src_op_bits)
        addr += 4
        mmio.write(addr, self.t2d_tdest)
        addr += 4
        mmio.write(addr, self.dst_op_bits)
        addr += 4
        mmio.write(addr, self.arith_op_bits)
        addr += 4
        mmio.write(addr, len(self.arith_op_tdest))
        addr += 4
        for elem in self.arith_op_tdest:
            mmio.write(addr, elem)
            addr += 4

        return addr

ACCL_DEFAULT_ARITH_CONFIG = {
    ('float16', 'float16', 'float16'): ACCLArithConfig(0, 16, 0, 16, [4], 16),
    ('float32', 'float16', 'float32'): ACCLArithConfig(0, 32, 1, 16, [0], 32),
    ('float32', 'float32', 'float32'): ACCLArithConfig(0, 32, 0, 32, [0], 32),
    ('float64', 'float64', 'float64'): ACCLArithConfig(0, 64, 0, 64, [1], 64),
    ('int32'  , 'int32'  , 'int32'  ): ACCLArithConfig(0, 32, 0, 32, [2], 32),
    ('int64'  , 'int64'  , 'int64'  ): ACCLArithConfig(0, 64, 0, 64, [3], 64),
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
    DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH =             13
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
    def __contains__(cls, item): 
        return item in [v.value for v in cls.__members__.values()] 

TAG_ANY = 0xFFFF_FFFF
EXCHANGE_MEM_OFFSET_ADDRESS= 0x1000
EXCHANGE_MEM_ADDRESS_RANGE = 0x1000
HOST_CTRL_ADDRESS_RANGE    = 0x800
RETCODE_OFFSET = 0x1FFC
IDCODE_OFFSET = 0x1FF8

class cclo(DefaultIP):
    """
    This class wrapps the common function of the collectives offload kernel
    """

    bindto = ["Xilinx:ACCL:ccl_offload:1.0"]

    def __init__(self, description):
        super().__init__(description=description)
        self._fullpath = description['fullpath']
        #define supported types and corresponding arithmetic config
        self.arith_config = {}
        self.arithcfg_addr = 0
        #define an empty list of RX spare buffers
        self.rx_buffer_spares = []
        self.rx_buffer_size = 0
        self.rx_buffers_adr = EXCHANGE_MEM_OFFSET_ADDRESS
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

    class dummy_address:
        def __init__(self, adr=0):
            self.device_address = adr

    def dump_exchange_memory(self):
        print("exchange mem:")
        num_word_per_line=4
        for i in range(0,EXCHANGE_MEM_ADDRESS_RANGE, 4*num_word_per_line):
            memory = []
            for j in range(num_word_per_line):
                memory.append(hex(self.read(EXCHANGE_MEM_OFFSET_ADDRESS+i+(j*4))))
            print(hex(EXCHANGE_MEM_OFFSET_ADDRESS + i), memory)

    def dump_host_control_memory(self):
        print("host control:")
        num_word_per_line=4
        for i in range(0,HOST_CTRL_ADDRESS_RANGE, 4*num_word_per_line):
            memory = []
            for j in range(num_word_per_line):
                memory.append(hex(self.read(i+(j*4))))
            print(hex(self.mmio.base_addr + i), memory)

    def deinit(self):
        print("Removing CCLO object at ",hex(self.mmio.base_addr))
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
            addr = self.arith_config[key].write(self.mmio, addr)

    def setup_rx_buffers(self, nbufs, bufsize, devicemem):
        addr = self.rx_buffers_adr
        self.rx_buffer_size = bufsize
        self.write(addr,nbufs)
        if not isinstance(devicemem, list):
            devicemem = [devicemem]
        for i in range(nbufs):
            #try to take a different bank where to put rank 
            devicemem_i = devicemem[ i % len(devicemem)]
           
            self.rx_buffer_spares.append(pynq.allocate((bufsize,), dtype=np.int8, target=devicemem_i))
            #program this buffer into the accelerator
            addr += 4
            self.write(addr, self.rx_buffer_spares[-1].physical_address & 0xffffffff)
            addr += 4
            self.write(addr, (self.rx_buffer_spares[-1].physical_address>>32) & 0xffffffff)
            addr += 4
            self.write(addr, bufsize)
            # clear remaining fields
            for _ in range(3,9):
                addr += 4
                self.write(addr, 0)

        self.communicators_addr = addr+4
        max_higher = 1
        self.utility_spare = pynq.allocate((bufsize*max_higher,), dtype=np.int8, target=devicemem[0])

        #Start irq-driven RX buffer scheduler and (de)packetizer
        #self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.reset_periph)
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.enable_irq)
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.enable_pkt)
        print("time taken to enqueue buffers", self.read(0x1FF4))
        #set segmentation size equal to buffer size
        self.set_dma_transaction_size(bufsize)
        self.set_max_dma_transaction_flight(10)
    
    def dump_rx_buffers_spares(self, nbufs=None):
        addr = self.rx_buffers_adr
        if nbufs is None:
            assert self.read(addr) == len(self.rx_buffer_spares)
            nbufs = len(self.rx_buffer_spares)
        print(f"CCLO address:{hex(self.mmio.base_addr)}")
        nbufs = min(len(self.rx_buffer_spares), nbufs)
        for i in range(nbufs):
            addr   += 4
            addrl   =self.read(addr)
            addr   += 4
            addrh   = self.read(addr)
            addr   += 4
            maxsize = self.read(addr)
            #assert self.read(addr) == self.rx_buffer_size
            addr   += 4
            dmatag  = self.read(addr)
            addr   += 4
            rstatus  = self.read(addr)
            addr   += 4
            rxtag   = self.read(addr)
            addr   += 4
            rxlen   = self.read(addr)
            addr   += 4
            rxsrc   = self.read(addr)
            addr   += 4
            seq     = self.read(addr)
            
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
                content = str(self.rx_buffer_spares[i].view(np.uint8))
            except Exception :
                content= "xxread failedxx"
            print(f"SPARE RX BUFFER{i}:\t ADDR: {hex(int(str(addrh)+str(addrl)))} \t STATUS: {status} \t OCCUPACY: {rxlen}/{maxsize} \t DMA TAG: {hex(dmatag)} \t  MPI TAG:{hex(rxtag)} \t SEQ: {seq} \t SRC:{rxsrc} \t content {content}")

    def call_async(self, scenario=CCLOp.nop, len=1, comm=0, root_src_dst=0, function=0, tag=TAG_ANY, arith_dtype=None, src_type=0, dst_type=0, addr_0=None, addr_1=None, addr_2=None, waitfor=[]):
        # no addresses, this is a config call
        if addr_0 is None and addr_2 is None:
            #no config needed
            arithcfg = 0
        # two- or three-address call, the common case
        elif addr_0 is not None and addr_2 is None:
            arithcfg = self.arith_config[(addr_0.dtype.name, addr_0.dtype.name, arith_dtype if arith_dtype is not None else addr_0.dtype.name)].addr
        elif addr_0 is None and addr_2 is not None:
            arithcfg = self.arith_config[(addr_2.dtype.name, addr_2.dtype.name, arith_dtype if arith_dtype is not None else addr_2.dtype.name)].addr
        elif addr_0 is not None and addr_2 is not None:
            arithcfg = self.arith_config[(addr_0.dtype.name, addr_2.dtype.name, arith_dtype if arith_dtype is not None else addr_0.dtype.name)].addr
        # set dummy addresses where needed
        if addr_0 is None:
            addr_0 = self.dummy_address()
        if addr_1 is None:
            addr_1 = self.dummy_address()
        if addr_2 is None:
            addr_2 = self.dummy_address()
        return self.start(scenario, len, comm, root_src_dst, function, tag, arithcfg, src_type, dst_type, addr_0, addr_1, addr_2, waitfor=waitfor)        

    def call_sync(self, scenario=CCLOp.nop, len=1, comm=0, root_src_dst=0, function=0, tag=TAG_ANY, arith_dtype=None, src_type=0, dst_type=0, addr_0=None, addr_1=None, addr_2=None):
        # no addresses, this is a config call
        if addr_0 is None and addr_2 is None:
            #no config needed
            arithcfg = 0
        # two- or three-address call, the common case
        elif addr_0 is not None and addr_2 is None:
            arithcfg = self.arith_config[(addr_0.dtype.name, addr_0.dtype.name, arith_dtype if arith_dtype is not None else addr_0.dtype.name)].addr
        elif addr_0 is None and addr_2 is not None:
            arithcfg = self.arith_config[(addr_2.dtype.name, addr_2.dtype.name, arith_dtype if arith_dtype is not None else addr_2.dtype.name)].addr
        elif addr_0 is not None and addr_2 is not None:
            arithcfg = self.arith_config[(addr_0.dtype.name, addr_2.dtype.name, arith_dtype if arith_dtype is not None else addr_0.dtype.name)].addr
        # set dummy addresses where needed
        if addr_0 is None:
            addr_0 = self.dummy_address()
        if addr_1 is None:
            addr_1 = self.dummy_address()
        if addr_2 is None:
            addr_2 = self.dummy_address()
        return self.call(scenario, len, comm, root_src_dst, function, tag, arithcfg, src_type, dst_type, addr_0, addr_1, addr_2)        

    def get_retcode(self):
        return self.read(RETCODE_OFFSET)

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
            error_msg = ErrorCode(retcode).name if retcode not in ErrorCode else f"UNKNOWN ERROR ({retcode})"
            raise Exception(f"CCLO @{hex(self.mmio.base_addr)}: during {label} {error_msg} you should consider resetting mpi_offload")
                

    def get_hwid(self):
        #TODO: add check
        return self.read(IDCODE_OFFSET) 

    def set_timeout(self, value, run_async=False, waitfor=[]):
        handle = self.call_async(scenario=CCLOp.config, len=value, function=CCLOCfgFunc.set_timeout, waitfor=waitfor)
        if run_async:
            return handle
        else:
            handle.wait()

    def start_profiling(self, run_async=False, waitfor=[]):
        handle = self.call_async(scenario=CCLOp.config, function=CCLOCfgFunc.start_profiling, waitfor=waitfor)
        if run_async:
            return handle
        else:
            handle.wait()

    def end_profiling(self, run_async=False, waitfor=[]):
        handle = self.call_async(scenario=CCLOp.config, function=CCLOCfgFunc.end_profiling, waitfor=waitfor)
        if run_async:
            return handle
        else:
            handle.wait()     

    def init_connection (self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, comm=self.communicators[comm_id]["addr"], function=CCLOCfgFunc.init_connection)
    
    @self_check_return_value
    def open_port(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, comm=self.communicators[comm_id]["addr"], function=CCLOCfgFunc.open_port)
    
    @self_check_return_value
    def open_con(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, comm=self.communicators[comm_id]["addr"], function=CCLOCfgFunc.open_con)
    
    @self_check_return_value
    def use_udp(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.use_udp_stack)
    
    @self_check_return_value
    def use_tcp(self, comm_id=0):
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.use_tcp_stack)   
    
    @self_check_return_value
    def set_dma_transaction_size(self, value=0):
        if value % 8 != 0:
            warnings.warn("ACCL: dma transaction must be divisible by 8 to use reduce collectives")
        elif value > self.rx_buffer_size:
            warnings.warn("ACCL: transaction size should be less or equal to configured buffer size!")
            return
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_dma_transaction_size, len=value)   
        self.segment_size = value
        print("time taken to start and stop timer", self.read(0x1FF4))

    @self_check_return_value
    def set_max_dma_transaction_flight(self, value=0):
     
        if value > 20:
            warnings.warn("ACCL: transaction size should be less or equal to configured buffer size!")
            return
        self.call_sync(scenario=CCLOp.config, function=CCLOCfgFunc.set_max_dma_transactions, len=value)   

    def configure_communicator(self, ranks, local_rank, vnx=False):
        assert len(self.rx_buffer_spares) > 0, "RX buffers unconfigured, please call setup_rx_buffers() first"
        if len(self.communicators) == 0:
            addr = self.communicators_addr
        else:
            addr = self.communicators[-1]["addr"]
        comm_address = EXCHANGE_MEM_OFFSET_ADDRESS + addr
        communicator = {"local_rank": local_rank, "addr": comm_address, "ranks": ranks, "inbound_seq_number_addr":[0 for _ in ranks], "outbound_seq_number_addr":[0 for _ in ranks], "session_addr":[0 for _ in ranks]}
        self.write(addr,len(ranks))
        addr += 4
        self.write(addr,local_rank)
        for i in range(len(ranks)):
            addr += 4
            #ip string to int conversion from here:
            #https://stackoverflow.com/questions/5619685/conversion-from-ip-string-to-integer-and-backward-in-python
            self.write(addr, int(ipaddress.IPv4Address(ranks[i]["ip"])))
            addr += 4
            #when using the UDP stack, write the rank number into the port register
            #the actual port is programmed into the stack itself
            if vnx:
                self.write(addr,i)
            else:
                self.write(addr,ranks[i]["port"])
            #leave 2 32 bit space for inbound/outbound_seq_number
            addr += 4
            self.write(addr,0)
            communicator["inbound_seq_number_addr"][i]  = addr
            addr +=4
            self.write(addr,0)
            communicator["outbound_seq_number_addr"][i] = addr
            #a 32 bit number is reserved for session id
            # sessions are initialized to 0xFFFFFFFF
            addr += 4
            self.write(addr, 0xFFFFFFFF)
            communicator["session_addr"][i] = addr
        self.communicators.append(communicator)
        self.arithcfg_addr = addr + 4
        
    def dump_communicator(self):
        if len(self.communicators) == 0:
            addr    = self.communicators_addr
        else:
            addr    = self.communicators[-1]["addr"] - EXCHANGE_MEM_OFFSET_ADDRESS
        nr_ranks    = self.read(addr)
        addr +=4
        local_rank  = self.read(addr)
        print(f"Communicator. local_rank: {local_rank} \t number of ranks: {nr_ranks}.")
        for i in range(nr_ranks):
            addr +=4
            #ip string to int conversion from here:
            #https://stackoverflow.com/questions/5619685/conversion-from-ip-string-to-integer-and-backward-in-python
            ip_addr_rank = str(ipaddress.IPv4Address(self.read(addr)))
            addr += 4
            #when using the UDP stack, write the rank number into the port register
            #the actual port is programmed into the stack itself
            port                = self.read(addr)
            #leave 2 32 bit space for inbound/outbound_seq_number
            addr += 4
            inbound_seq_number  = self.read(addr)
            addr +=4
            outbound_seq_number = self.read(addr)
            #a 32 bit integer is dedicated to session id 
            addr += 4
            session = self.read(addr)
            print(f"> rank {i} (ip {ip_addr_rank}:{port} ; session {session}) : <- inbound_seq_number {inbound_seq_number}, -> outbound_seq_number {outbound_seq_number}")
   

    @self_check_return_value
    def nop(self, run_async=False, waitfor=[]):
        #calls the accelerator with no work. Useful for measuring call latency
        handle = self.call_async(scenario=CCLOp.nop, waitfor=waitfor)
        if run_async:
            return handle 
        else:
            handle.wait()

    @self_check_return_value
    def send(self, comm_id, srcbuf, dst, tag=TAG_ANY, from_fpga=False, run_async=False, waitfor=[]):
        if srcbuf.nbytes == 0:
            warnings.warn("zero size buffer")
            return
        if not from_fpga:
            srcbuf.sync_to_device()
        handle = self.call_async(scenario=CCLOp.send, len=srcbuf.nbytes, comm=self.communicators[comm_id]["addr"], root_src_dst=dst, tag=tag, addr_0=srcbuf, waitfor=waitfor)
        if run_async:
            return handle 
        else:
            handle.wait()
    
    @self_check_return_value
    def recv(self, comm_id, dstbuf, src, tag=TAG_ANY, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if dstbuf.nbytes == 0:
            warnings.warn("zero size buffer")
            return
        handle = self.call_async(scenario=CCLOp.recv, len=dstbuf.nbytes, comm=self.communicators[comm_id]["addr"], root_src_dst=src, tag=tag, addr_0=dstbuf, waitfor=waitfor)
        if run_async:
            return handle
        else:
            handle.wait()
        if not to_fpga:
            dstbuf.sync_from_device()

    @self_check_return_value
    def copy(self, srcbuf, dstbuf,  from_fpga=False,  to_fpga=False, run_async=False, waitfor=[], arith_dtype=None):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if srcbuf.nbytes == 0:
            warnings.warn("zero size buffer")
            return
        # performs dstbuf = srcbuf
        if not from_fpga:
            srcbuf.sync_to_device()
        import pdb; pdb.set_trace()
        handle = self.call_async(scenario=CCLOp.copy, len=srcbuf.size, addr_0=srcbuf, addr_2=dstbuf, waitfor=waitfor, arith_dtype=arith_dtype)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            dstbuf.sync_from_device()

    @self_check_return_value
    def sum(self, func, val1, val2, sum, val1_from_fpga=False, val2_from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        assert val1.nbytes != 0 or val2.nbytes != 0, "Zero-size input buffer"
        assert val1.nbytes == val2.nbytes, "Unequal sized operands"
        # TODO: check datatype support
        # performs acc = val + acc
        if not val1_from_fpga:
            val1.sync_to_device()
        if not val2_from_fpga:
            val2.sync_to_device()
        handle = self.call_async(scenario=CCLOp.sum, len=val1.nbytes, function=func, addr_0=val1, addr_1=val2, addr_2=sum, waitfor=waitfor)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            acc.sync_from_device()
    
    @self_check_return_value
    def external_stream_kernel(self, src_buf, dst_buf, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        if not to_fpga and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if src_buf.nbytes <= 4:
            warnings.warn("size of buffer not compatible")
            return

        if not from_fpga:
            src_buf.sync_to_device()

        handle = self.call_async(scenario=CCLOp.ext_stream_krnl, len=src_buf.nbytes, addr_0=src_buf, addr_1=dst_buf, waitfor=waitfor)
        if run_async:
            return handle
        
        handle.wait()
        if not to_fpga:
            dst_buf.sync_from_device()

    @self_check_return_value
    def bcast(self, comm_id, buf, root, from_fpga=False, to_fpga=False, run_async=False, waitfor=[]):
        comm = self.communicators[comm_id]
        is_root = comm["local_rank"] == root
        if not to_fpga and not(is_root) and run_async:
            warnings.warn("ACCL: async run returns data on FPGA, user must sync_from_device() after waiting")
        if buf.nbytes == 0:
            warnings.warn("zero size buffer")
            return
        # sync the transmit source in one go
        if not from_fpga and is_root:
            buf.sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.bcast, len=buf.nbytes, comm=self.communicators[comm_id]["addr"], root_src_dst=root, addr_0=buf, waitfor=waitfor)]
        
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

        prevcall = [self.call_async(scenario=CCLOp.scatter, len=rbuf[0:count].nbytes, comm=comm["addr"], root_src_dst=root, addr_0=sbuf, addr_1=rbuf[0:count], waitfor=waitfor)]

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
            
        prevcall = [self.call_async(scenario=CCLOp.gather, len=rbuf[0:count].nbytes, comm=comm["addr"], root_src_dst=root, addr_0=sbuf, addr_1=rbuf, waitfor=waitfor)]
            
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

        prevcall = [self.call_async(scenario=CCLOp.allgather, len=rbuf[0:count].nbytes, comm=comm["addr"], addr_0=sbuf, addr_1=rbuf, waitfor=waitfor)]
            
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
        # TODO: check datatype support
        assert count % sbuf.itemsize == 0, "Count not a multiple of input element size"

        comm        = self.communicators[comm_id]
        p           = len(comm["ranks"])
        local_rank  = comm["local_rank"]

        if not from_fpga:
            sbuf[0:count].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.reduce, len=count, comm=self.communicators[comm_id]["addr"], root_src_dst=root, function=func, addr_0=sbuf, addr_1=rbuf, waitfor=waitfor)]

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
        # TODO: check datatype support
        assert count % sbuf.itemsize == 0, "Count not a multiple of input element size"

        if not from_fpga:
            sbuf[0:count].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.allreduce, len=count, comm=self.communicators[comm_id]["addr"], function=func, addr_0=sbuf, addr_1=rbuf, waitfor=waitfor)]

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

        # TODO: check datatype support
        assert count % sbuf.itemsize == 0, "Count not a multiple of input element size"

        comm        = self.communicators[comm_id]
        p           = len(comm["ranks"])
        local_rank  = comm["local_rank"]

        if not from_fpga:
            sbuf[0:count*p].sync_to_device()

        prevcall = [self.call_async(scenario=CCLOp.reduce_scatter, len=count, comm=self.communicators[comm_id]["addr"], function=func, addr_0=sbuf, addr_1=rbuf, waitfor=waitfor)]

        if run_async:
            return prevcall[0]
        
        prevcall[0].wait()
        if not to_fpga and local_rank == root:
            rbuf[0:count*p].sync_from_device()
