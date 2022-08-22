/*******************************************************************************
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

#include "cclo_bfm.h"
#include "zmq_client.h"
#include <stdexcept>
#include "constants.hpp"

CCLO_BFM::CCLO_BFM(unsigned int zmqport, unsigned int local_rank, unsigned int world_size,  unsigned int krnl_dest,
            Stream<command_word> &callreq, Stream<command_word> &callack,
            Stream<stream_word> &m_krnl, Stream<stream_word> &s_krnl,
            int target_ctrl_stream) : 
            callreq(callreq), callack(callack), m_krnl(m_krnl), s_krnl(s_krnl), target_ctrl_stream(target_ctrl_stream) {
    //create ZMQ context
    debug("CCLO BFM connecting to ZMQ on starting port " + std::to_string(zmqport) + " for rank " + std::to_string(local_rank));
    ctx = zmq_client_intf(zmqport, local_rank, world_size, krnl_dest);
    debug("CCLO BFM connected");
}

//utility function, finds the registered SimBuffer
//to which a given address belongs, and performs sanity checks
unsigned int CCLO_BFM::find_registered_buffer(uint64_t adr){
    for(int i=0; i<this->buffers.size(); i++){
        SimBuffer *tmp = buffers.at(i);
        uint64_t tmp_start_adr = (uint64_t)tmp->byte_array();
        uint64_t tmp_end_adr = tmp_start_adr + tmp->size();
        //check the start address is within the range of tmp
        if(adr < tmp_start_adr || adr >= tmp_end_adr) continue;
        //return this buffer's index
        return i;
    }
    throw std::invalid_argument("No registered buffer found for this request");
}

void CCLO_BFM::push_cmd(){
    unsigned int scenario, tag, count, comm, root_src_dst, function, arithcfg_addr, compression_flags, stream_flags;
    uint64_t addr_0, addr_1,addr_2;
    while(!finalize){
        scenario = callreq.Pop().data;
        tag = callreq.Pop().data;
        count = callreq.Pop().data;
        comm = callreq.Pop().data;
        root_src_dst = callreq.Pop().data;
        function = callreq.Pop().data;
        arithcfg_addr = callreq.Pop().data;
        compression_flags = callreq.Pop().data;
        stream_flags = callreq.Pop().data;
        addr_0 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        addr_1 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        addr_2 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        SimBuffer *buf;
        std::vector<SimBuffer*> bufargs;
        if(count > 0 && addr_0 != 0){
            buf = this->buffers(find_registered_buffer(addr_0));
            buf->sync_to_device();
            bufargs.push_back(buf);
        }
        if(count > 0 && addr_1 != 0){
            buf = this->buffers(find_registered_buffer(addr_1));
            buf->sync_to_device();
            bufargs.push_back(buf);
        }
        if(count > 0 && addr_2 != 0){
            buf = this->buffers(find_registered_buffer(addr_2));
            buf->sync_to_device();
            bufargs.push_back(buf);
        }
        this->buf_args.Push(bufargs);
        zmq_client_startcall(ctx, scenario, tag, count,
                            comm, root_src_dst, function,
                            arithcfg_addr, compression_flags, stream_flags,
                            addr_0, addr_1, addr_2, target_ctrl_stream);
    }
}

void CCLO_BFM::pop_sts(){
    std::vector<SimBuffer*> bufargs;
    while(!finalize){
        //wait for simulator/emulator to finish executing
        zmq_client_retcall(ctx);
        //get buffers and sync them back
        bufargs = this->buf_args.Pop();
        for(int i=0; i<bufargs.size(); i++){
            bufargs.at(i)->sync_from_device();
        }
        //signal completion to kernel
        callack.Push({.data=0, .last=1});
    }
}

void CCLO_BFM::push_krnl(){
    while(!finalize){
        ap_axiu<512,0,0,8> tmp;
        vector<uint8_t> vec;
        do{
            tmp = s_krnl.Pop();
            for(int i=0; i<512/8; i++){
                vec.push_back(tmp.data((i+1)*8-1,i*8));
            }
        } while(tmp.last == 0);
        zmq_client_strmwrite(ctx, vec);
    }
}

void CCLO_BFM::pop_krnl(){
    while(!finalize){
        ap_axiu<512,0,0,8> tmp;
        vector<uint8_t> vec;
        zmq_client_strmread(ctx, vec);
        int idx;
        do{
            for(i=0; i<512/8, idx<vec.size(); i++){
                tmp.data((i+1)*8-1,i*8) = vec.at(idx++);
            }
            tmp.last = (idx == vec.size());
            m_krnl.Push(tmp);
        } while(tmp.last == 0);
    }
}

void CCLO_BFM::run(){
    //start threads
    finalize = false;
    //command interface threads
    std::thread t1(push_cmd);
    threads.push_back(t1);
    std::thread t2(pop_sts);
    threads.push_back(t2);
    //kernel interface threads
    std::thread t3(push_krnl);
    threads.push_back(t3);
    std::thread t4(pop_krnl);
    threads.push_back(t4);
}

void CCLO_BFM::stop(){
    finalize = true;
    for(int i=0; i<threads.size(); i++){
        threads.at(i).join();
    }
}

void CCLO_BFM::register_buffer(SimBuffer* buf){
    this->buffers.push_back(buf);
}
