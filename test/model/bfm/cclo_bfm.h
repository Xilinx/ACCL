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

/**
 * @brief Class providing a bus-functional model (at HLS Stream level) of the ACCL CCLO kernel. Connects to the emulator/simulator.
 *
 */
class CCLO_BFM{
    private:
        zmq_intf_context zmq_ctx;
        hlslib::Stream<command_word> &callreq;
        hlslib::Stream<command_word> &callack;
        hlslib::Stream<stream_word> &data_cclo2krnl;
        hlslib::Stream<stream_word> &data_krnl2cclo;

        bool finalize;
        std::vector<std::thread> threads;
        int target_ctrl_stream;

        //hlslib::Stream<std::vector<ACCL::SimBuffer *>> buf_args;
        //std::vector<ACCL::SimBuffer *> buffers;

    public:
        /**
         * @brief Construct a new CCLO_BFM object
         *
         * @param zmqport Number of port which connects to the ACCL emulator/simulator
         * @param local_rank ID of local rank
         * @param world_size Total number of ranks
         * @param krnl_dest A vector of CCLO data port destination IDs to which to subscribe
         * @param callreq HLS Stream for call commands issued by HLS functions
         * @param callreq HLS Stream for call responses to HLS functions
         * @param data_cclo2krnl HLS Stream for data provided by the (emulated) CCLO to HLS functions
         * @param data_krnl2cclo HLS Stream for data provided by the HLS functions to the (emulated) CCLO
         * @param target_ctrl_stream Control stream to use inside emulator/simulator. Do not change.
         */
        CCLO_BFM(unsigned int zmqport, unsigned int local_rank, unsigned int world_size,  const std::vector<unsigned int>& krnl_dest,
                    hlslib::Stream<command_word> &callreq, hlslib::Stream<command_word> &callack,
                    hlslib::Stream<stream_word> &data_cclo2krnl, hlslib::Stream<stream_word> &data_krnl2cclo, int target_ctrl_stream=2);

        /**
        * @brief Deconstructer of the CCLO_BFM object
        *
        */
        ~CCLO_BFM() {}

        /**
         * @brief Start BFM
         *
         */
        void run();

        /**
         * @brief Stop BFM
         *
         */
        void stop();
        //void register_buffer(ACCL::SimBuffer *buf)

    private:
        void push_cmd();
        void pop_sts();
        void pop_data();
        void push_data();
        //unsigned int find_registered_buffer(uint64_t adr);

};
