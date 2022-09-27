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
#include <iostream>
#include <mutex>

std::mutex call_ctr_mutex;
unsigned int call_ctr;

CCLO_BFM::CCLO_BFM(unsigned int zmqport, unsigned int local_rank, unsigned int world_size,  const std::vector<unsigned int>& krnl_dest,
            hlslib::Stream<command_word> &callreq, hlslib::Stream<command_word> &callack,
            hlslib::Stream<stream_word> &data_cclo2krnl, hlslib::Stream<stream_word> &data_krnl2cclo,
            int target_ctrl_stream) :
            callreq(callreq), callack(callack), data_cclo2krnl(data_cclo2krnl), data_krnl2cclo(data_krnl2cclo), target_ctrl_stream(target_ctrl_stream) {
    //create ZMQ context
    std::cout << "CCLO BFM connecting to ZMQ on starting port " + std::to_string(zmqport) + " for rank " + std::to_string(local_rank) << std::endl;
    zmq_ctx = zmq_client_intf(zmqport, local_rank, krnl_dest, world_size);
    std::cout << "CCLO BFM connected" << std::endl;
}

//TODO: utility function, finds the registered SimBuffer
//to which a given address belongs, and performs sanity checks
// unsigned int CCLO_BFM::find_registered_buffer(uint64_t adr){
//     for(int i=0; i<this->buffers.size(); i++){
//         SimBuffer *tmp = buffers.at(i);
//         uint64_t tmp_start_adr = (uint64_t)tmp->byte_array();
//         uint64_t tmp_end_adr = tmp_start_adr + tmp->size();
//         //check the start address is within the range of tmp
//         if(adr < tmp_start_adr || adr >= tmp_end_adr) continue;
//         //return this buffer's index
//         return i;
//     }
//     throw std::invalid_argument("No registered buffer found for this request");
// }

void CCLO_BFM::push_cmd(){
    unsigned int scenario, tag, count, comm, root_src_dst, function, arithcfg_addr, compression_flags, stream_flags;
    uint64_t addr_0, addr_1,addr_2;
    while(!finalize){
        if(callreq.IsEmpty()) continue;
        scenario = callreq.Pop().data;
        count = callreq.Pop().data;
        comm = callreq.Pop().data;
        root_src_dst = callreq.Pop().data;
        function = callreq.Pop().data;
        tag = callreq.Pop().data;
        arithcfg_addr = callreq.Pop().data;
        compression_flags = callreq.Pop().data;
        stream_flags = callreq.Pop().data;
        addr_0 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        addr_1 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        addr_2 = (uint64_t)callreq.Pop().data | ((uint64_t)callreq.Pop().data)<<32;
        //TODO: use addresses to look up registered buffers and sync them
        // SimBuffer *buf;
        // std::vector<SimBuffer*> bufargs;
        // if(count > 0 && addr_0 != 0){
        //     buf = this->buffers(find_registered_buffer(addr_0));
        //     buf->sync_to_device();
        //     bufargs.push_back(buf);
        // }
        // if(count > 0 && addr_1 != 0){
        //     buf = this->buffers(find_registered_buffer(addr_1));
        //     buf->sync_to_device();
        //     bufargs.push_back(buf);
        // }
        // if(count > 0 && addr_2 != 0){
        //     buf = this->buffers(find_registered_buffer(addr_2));
        //     buf->sync_to_device();
        //     bufargs.push_back(buf);
        // }
        // this->buf_args.Push(bufargs);
        zmq_client_startcall(&zmq_ctx, scenario, tag, count,
                            comm, root_src_dst, function,
                            arithcfg_addr, compression_flags, stream_flags,
                            addr_0, addr_1, addr_2, target_ctrl_stream);
        call_ctr_mutex.lock();
        call_ctr++;
        call_ctr_mutex.unlock();
    }
}

void CCLO_BFM::pop_sts(){
    //std::vector<SimBuffer*> bufargs;
    while(!finalize){
        if(call_ctr == 0) continue;
        //wait for simulator/emulator to finish executing
        zmq_client_retcall(&zmq_ctx, target_ctrl_stream);
        //TODO: get buffers and sync them back
        // bufargs = this->buf_args.Pop();
        // for(int i=0; i<bufargs.size(); i++){
        //     bufargs.at(i)->sync_from_device();
        // }
        //signal completion to kernel
        callack.Push({.data=0, .last=1});
        call_ctr_mutex.lock();
        call_ctr--;
        call_ctr_mutex.unlock();
    }
}

void CCLO_BFM::push_data(){
    while(!finalize){
        stream_word tmp;
        std::vector<uint8_t> vec;
        if(data_krnl2cclo.IsEmpty()) continue;
        do{
            tmp = data_krnl2cclo.Pop();
            for(int i=0; i<DATA_WIDTH/8; i++){
                vec.push_back(tmp.data((i+1)*8-1,i*8));
            }
        } while(tmp.last == 0);
        zmq_client_strmwrite(&zmq_ctx, vec);
    }
}

void CCLO_BFM::pop_data(){
    while(!finalize){
        stream_word tmp;
        std::vector<uint8_t> vec;
        vec = zmq_client_strmread(&zmq_ctx, true);
        if(vec.size() == 0) continue;
        unsigned int idx;
        do{
            for(unsigned int i=0; i<DATA_WIDTH/8 && idx<vec.size(); i++){
                tmp.data((i+1)*8-1,i*8) = vec.at(idx++);
            }
            tmp.last = (idx == vec.size());
            data_cclo2krnl.Push(tmp);
        } while(tmp.last == 0);
    }
}

void CCLO_BFM::run(){
    //start threads
    finalize = false;
    call_ctr = 0;
    //command interface threads
    std::thread t1(&CCLO_BFM::push_cmd, this);
    threads.push_back(move(t1));
    std::thread t2(&CCLO_BFM::pop_sts, this);
    threads.push_back(move(t2));
    //kernel interface threads
    std::thread t3(&CCLO_BFM::push_data, this);
    threads.push_back(move(t3));
    std::thread t4(&CCLO_BFM::pop_data, this);
    threads.push_back(move(t4));
}

void CCLO_BFM::stop(){
    finalize = true;
    for(unsigned int i=0; i<threads.size(); i++){
        threads.at(i).join();
    }
    std::cout << "CCLO BFM stopped" << std::endl;
}

//TODO: a register_buffer function that works with any SimBuffer
// void CCLO_BFM::register_buffer(SimBuffer* buf){
//     this->buffers.push_back(buf);
// }
