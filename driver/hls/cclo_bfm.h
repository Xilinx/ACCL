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

#pragma once
#include "zmq_client.h"
#include "accl_hls.h"
#include "simbuffer.hpp"
#include <vector>

class CCLO_BFM{
    private:
        zmq_intf_context zmq_ctx;
        hlslib::Stream<command_word> &callreq;
        hlslib::Stream<command_word> &callack;
        hlslib::Stream<stream_word> &m_krnl;
        hlslib::Stream<stream_word> &s_krnl;

        bool finalize;
        std::vector<std::thread> threads;
        int target_ctrl_stream;

        //hlslib::Stream<std::vector<ACCL::SimBuffer *>> buf_args; 
        //std::vector<ACCL::SimBuffer *> buffers;

    public:
        CCLO_BFM(unsigned int zmqport, unsigned int local_rank, unsigned int world_size,  unsigned int krnl_dest,
                    hlslib::Stream<command_word> &callreq, hlslib::Stream<command_word> &callack,
                    hlslib::Stream<stream_word> &m_krnl, hlslib::Stream<stream_word> &s_krnl, int target_ctrl_stream=2);
        void run();
        void stop();
        //void register_buffer(ACCL::SimBuffer *buf)

    private:
        void push_cmd();
        void pop_sts();
        void pop_krnl();
        void push_krnl();
        //unsigned int find_registered_buffer(uint64_t adr);

};