/*******************************************************************************
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
#
*******************************************************************************/

#include <accl.hpp>
#include <cstdlib>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include <accl_hls.h>
#include <cclo_bfm.h>
#include <xrt/xrt_device.h>
#include <iostream>

using namespace ACCL;

int rank, size;

//hls-synthesizable function performing
//an elementwise increment on fp32 data in src 
//followed by a put to another rank, then
//writing result in dst
void vadd_s2s(
    float *src,
    float *dst,
    int count,
    unsigned int destination,
    //parameters pertaining to CCLO config
    ap_uint<32> comm_adr, 
    ap_uint<32> dpcfg_adr,
    //streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
){
    //set up interfaces
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, comm_adr, dpcfg_adr, 0, 3);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    //read data from src, increment it, 
    //and push the result into the CCLO stream
    ap_uint<512> tmpword;
    int rd_count = count;
    while(rd_count > 0){
        //read 16 floats into a 512b vector
        for(int i=0; (i<16) && (rd_count>0); i++){
            float inc = src[i]+1;
            tmpword(i*32,(i+1)*32-1) = *reinterpret_cast<ap_uint<32>*>(&inc);
            rd_count--;
        }
        //send the vector to cclo
        data.push(tmpword, 0);
    }
    //send command to CCLO
    //we're passing src as source 
    //because we're streaming data, the address will be ignored
    //we're passing 9 as stream ID, because IDs 0-8 are reserved
    accl.stream_put(count, 9, destination, (ap_uint<64>)src);
    //pull data from CCLO and write it to dst
    int wr_count = count;
    while(wr_count > 0){
        //read vector from CCLO
        tmpword = data.pull().data;
        //read from the 512b vector into 16 floats
        for(int i=0; (i<16) && (wr_count>0); i++){
            ap_uint<32> val = tmpword(i*32,(i+1)*32-1);
            dst[i] = *reinterpret_cast<float*>(&val);
            wr_count--;
        }
    }
}

struct options_t {
    int start_port;
    unsigned int rxbuf_size;
    unsigned int count;
    unsigned int dest;
    unsigned int nruns;
    unsigned int device_index;
};

void run_test(options_t options) {
    std::vector<rank_t> ranks = {};
    for (int i = 0; i < size; ++i) {
        rank_t new_rank = {"127.0.0.1", options.start_port + i, i, options.rxbuf_size};
        ranks.emplace_back(new_rank);
    }

    std::unique_ptr<ACCL::ACCL> accl;
    accl = std::make_unique<ACCL::ACCL>(ranks, rank, options.start_port,
                                            networkProtocol::TCP, 16,
                                            options.rxbuf_size);

    accl->set_timeout(1e8);
    std::cout << "Host-side CCLO initialization finished" << std::endl;

    // barrier here to make sure all the devices are configured before testing
    MPI_Barrier(MPI_COMM_WORLD);

    //run test here:
    //initialize a CCLO BFM and streams as needed
    hlslib::Stream<command_word> callreq, callack;
    hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
    std::vector<unsigned int> dest = {9};
    CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
    cclo.run();
    std::cout << "CCLO BFM started" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    //allocate float arrays for the HLS function to use
    float src[options.count], dst[options.count];
    for(int i=0; i<options.count; i++){
        src[i] = 0;
    }
    //run the hls function, using the global communicator
    vadd_s2s(   src, dst, options.count, 
                (rank+1)%size,
                accl->get_communicator_adr(), 
                accl->get_arithmetic_config_addr({dataType::float32, dataType::float32}), 
                callreq, callack, 
                data_krnl2cclo, data_cclo2krnl);
    //check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i<options.count; i++){
        err_count += (dst[i] != 1);
    }
    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    //clean up
    cclo.stop();
    accl->deinit();
}


options_t parse_options(int argc, char *argv[]) {
    TCLAP::CmdLine cmd("Test ACCL C++ driver");
    TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
                                            "How many times to run each test",
                                            false, 1, "positive integer");
    cmd.add(nruns_arg);
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "s", "start-port", "Start of range of ports usable for sim", false, 5500,
        "positive integer");
    cmd.add(start_port_arg);
    TCLAP::ValueArg<uint32_t> count_arg("c", "count", "How many bytes per buffer",
                                        false, 16, "positive integer");
    cmd.add(count_arg);
    TCLAP::ValueArg<uint32_t> bufsize_arg("b", "rxbuf-size",
                                            "How many KB per RX buffer", false, 1,
                                            "positive integer");
    cmd.add(bufsize_arg);

    try {
        cmd.parse(argc, argv);
    } catch (std::exception &e) {
        if (rank == 0) {
        std::cout << "Error: " << e.what() << std::endl;
        }

        MPI_Finalize();
        exit(1);
    }

    options_t opts;
    opts.start_port = start_port_arg.getValue();
    opts.count = count_arg.getValue();
    opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
    opts.nruns = nruns_arg.getValue();
    return opts;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    options_t options = parse_options(argc, argv);

    std::ostringstream stream;
    stream << "rank " << rank << " size " << size << std::endl;
    std::cout << stream.str();

    run_test(options);

    MPI_Finalize();
    return 0;
}
