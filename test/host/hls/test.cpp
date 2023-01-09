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
#include <accl_network_utils.hpp>
#include <cstdlib>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include <xrt/xrt_device.h>
#include <iostream>

#include "vadd_put.h"
#include "cclo_bfm.h"

using namespace ACCL;
using namespace accl_network_utils;

int rank, size;

struct options_t {
    int start_port;
    unsigned int rxbuf_size;
    unsigned int segment_size;
    unsigned int count;
    unsigned int dest;
    unsigned int nruns;
    unsigned int device_index;
    bool udp;
    bool hardware;
    bool rsfec;
    std::string xclbin;
    std::string config_file;
};

void test_vadd_put(ACCL::ACCL &accl, xrt::device &device, options_t options) {
    //run test here:

    //allocate float arrays for the HLS function to use
    float src[options.count], dst[options.count];
    for(int i=0; i<options.count; i++){
        src[i] = 1.0*(options.count*rank+i);
    }

    if (options.hardware) {
        auto vadd_ip = xrt::kernel(device, device.get_xclbin_uuid(), "vadd_put:{vadd_" + std::to_string(rank) + "_0}",
                        xrt::kernel::cu_access_mode::exclusive);
        //need to use XRT API because vadd kernel might use different HBM banks than ACCL
        auto src_bo = xrt::bo(device, sizeof(float)*options.count, vadd_ip.group_id(0));
        auto dst_bo = xrt::bo(device, sizeof(float)*options.count, vadd_ip.group_id(1));

        src_bo.write(src);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = vadd_ip(src_bo, dst_bo, options.count, (rank+1)%size, accl.get_communicator_addr(),
                    accl.get_arithmetic_config_addr({dataType::float32, dataType::float32}));
        run.wait(10000);

        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        dst_bo.read(dst);
    } else {
        //initialize a CCLO BFM and streams as needed
        hlslib::Stream<command_word> callreq, callack;
        hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
        std::vector<unsigned int> dest = {9};
        CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
        cclo.run();
        std::cout << "CCLO BFM started" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        //run the hls function, using the global communicator
        vadd_put(   src, dst, options.count,
                    (rank+1)%size,
                    accl.get_communicator_addr(),
                    accl.get_arithmetic_config_addr({dataType::float32, dataType::float32}),
                    callreq, callack,
                    data_krnl2cclo, data_cclo2krnl);
        //stop the BFM
        cclo.stop();
    }

    //check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i<options.count; i++){
        float expected = 1.0*(options.count*((rank+size-1)%size)+i) + 1;
        if(dst[i] != expected){
            err_count++;
            std::cout << "Mismatch at [" << i << "]: got " << dst[i] << " vs expected " << expected << std::endl;
        }
    }

    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_loopback_local_res(ACCL::ACCL& accl, options_t options) {

    //run test here:
    //initialize a CCLO BFM and streams as needed
    hlslib::Stream<command_word> callreq, callack;
    hlslib::Stream<stream_word, 512> data_cclo2krnl("cclo2krnl"), data_krnl2cclo("krnl2cclo");
    int stream_id = 9;
    std::vector<unsigned int> dest = {stream_id};
    CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
    cclo.run();
    std::cout << "CCLO BFM started" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    stream_word in;
    in.dest = stream_id;
    in.last = 1;
    in.keep = -1;
    in.data = rank;
    for (int i=0; i < (options.count+15)/16; i++) {
        data_krnl2cclo.write(in);
    }
    accl.stream_put(dataType::int32, options.count, rank, stream_id);

    //loop back data (divide count by 16 and round up to get number of stream words)
    std::vector<stream_word> recv_words;
    for (int i=0; i < (options.count+15)/16; i++) {
        recv_words.push_back(data_cclo2krnl.read());
    }

    //check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i<(options.count+15)/16; i++){
        err_count += (recv_words[i].data != rank);
    }

    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    //clean up
    cclo.stop();
}

void test_loopback(ACCL::ACCL& accl, options_t options, unsigned char stream_id) {

    //run test here:
    //initialize a CCLO BFM and streams as needed
    hlslib::Stream<command_word> callreq, callack;
    hlslib::Stream<stream_word, 512> data_cclo2krnl("cclo2krnl"), data_krnl2cclo("krnl2cclo");
    std::vector<unsigned int> dest = {stream_id};
    CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
    cclo.run();
    std::cout << "CCLO BFM started" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    //allocate float arrays for the HLS function to use
    auto src_buffer = accl.create_buffer<int>(options.count, ACCL::dataType::int32, 0);
    auto dst_buffer = accl.create_buffer<int>(options.count, ACCL::dataType::int32, 0);
    for(int i=0; i<options.count; i++){
        src_buffer->buffer()[i] = rank;
        dst_buffer->buffer()[i] = 0;
    }

    int init_rank = ((rank - 1 + size) % size);
    int loopback_rank = ((rank + 1 ) % size);

    accl.send(*src_buffer, options.count, loopback_rank, stream_id);
    accl.recv(dataType::int32, options.count, init_rank, stream_id, ACCL::GLOBAL_COMM);

    //loop back data (divide count by 16 and round up to get number of stream words)
    for (int i=0; i < (options.count+15)/16; i++) {
        data_krnl2cclo.write(data_cclo2krnl.read());
    }

    accl.send(dataType::int32, options.count, init_rank, stream_id, ACCL::GLOBAL_COMM);

    accl.recv(*dst_buffer, options.count,  loopback_rank, stream_id);
    //check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i<options.count; i++){
        err_count += (dst_buffer->buffer()[i] != rank);
    }

    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    //clean up
    cclo.stop();
}


void test_reduce_stream(ACCL::ACCL& accl, options_t options) {

    //run test here:
    //initialize a CCLO BFM and streams as needed
    hlslib::Stream<command_word> callreq, callack;
    hlslib::Stream<stream_word, 512> data_cclo2krnl("cclo2krnl"), data_krnl2cclo("krnl2cclo");
    std::vector<unsigned int> dest = {0};
    CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
    cclo.run();
    std::cout << "CCLO BFM started" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    //allocate int arrays for the HLS function to use
    auto src_buffer = accl.create_buffer<int>(options.count, ACCL::dataType::int32, 0);
    auto dst_buffer = accl.create_buffer<int>(options.count, ACCL::dataType::int32, 0);
    for(int i=0; i<options.count; i++){
        src_buffer->buffer()[i] = i;
        dst_buffer->buffer()[i] = 0;
    }

    int init_rank = ((rank - 1 + size) % size);
    int loopback_rank = ((rank + 1 ) % size);

    stream_word tmp;
    tmp.dest = 0;
    tmp.last = 1;
    tmp.keep = -1;

    //load data into stream
    for (int i=0; i < options.count; i++) {
        int idx = i % (DATA_WIDTH/32);
        tmp.data((idx+1)*32-1,idx*32) = src_buffer->buffer()[i];
        if(idx == (DATA_WIDTH/32-1)){
            data_krnl2cclo.write(tmp);
        }
    }

    accl.reduce(dataType::int32, dataType::int32, options.count, 0, reduceFunction::SUM);

    unsigned int err_count = 0;
    if(rank == 0){
        //read back data
        for (int i=0; i < options.count; i++) {
            int idx = i % (DATA_WIDTH/32);
            if(idx == 0)
                tmp = data_cclo2krnl.read();
            dst_buffer->buffer()[i] = tmp.data((idx+1)*32-1,idx*32);
        }

        //check HLS function outputs
        for(int i=0; i<options.count; i++){
            err_count += (dst_buffer->buffer()[i] != i*size);
        }
    }

    std::cout << "Test finished with " << err_count << " errors" << std::endl;
    //clean up
    cclo.stop();
}


options_t parse_options(int argc, char *argv[]) {
    TCLAP::CmdLine cmd("Test ACCL C++ driver");
    TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
                                            "How many times to run each test",
                                            false, 1, "positive integer");
    cmd.add(nruns_arg);
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "p", "start-port", "Start of range of ports usable for sim", false, 5500,
        "positive integer");
    cmd.add(start_port_arg);
    TCLAP::ValueArg<uint32_t> count_arg("s", "count", "How many bytes per buffer",
                                        false, 16, "positive integer");
    cmd.add(count_arg);
    TCLAP::ValueArg<uint32_t> bufsize_arg("b", "rxbuf-size",
                                            "How many KB per RX buffer", false, 1,
                                            "positive integer");
    cmd.add(bufsize_arg);
    TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP backend", cmd, false);
    TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd, false);
    TCLAP::ValueArg<std::string> xclbin_arg(
        "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
        "accl.xclbin", "file");
    cmd.add(xclbin_arg);
    TCLAP::ValueArg<uint16_t> device_index_arg(
        "i", "device-index", "device index of FPGA if hardware mode is used",
        false, 0, "positive integer");
    cmd.add(device_index_arg);
    TCLAP::ValueArg<std::string> config_arg(
        "c", "config", "Config file containing IP mapping",
        false, "", "JSON file");
    TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd,
                               false);
    cmd.add(config_arg);

    try {
        cmd.parse(argc, argv);
        if(hardware_arg.getValue()) {
            if(udp_arg.getValue()) {
                throw std::runtime_error("Hardware run only supported on axis3x.");
            }
        }
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
    opts.segment_size = opts.rxbuf_size;
    opts.nruns = nruns_arg.getValue();
    opts.udp = udp_arg.getValue();
    opts.hardware = hardware_arg.getValue();
    opts.xclbin = xclbin_arg.getValue();
    opts.device_index = device_index_arg.getValue();
    opts.config_file = config_arg.getValue();
    opts.rsfec = rsfec_arg.getValue();
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

    acclDesign design = acclDesign::AXIS3x;
    std::vector<rank_t> ranks;
    if (options.config_file == "") {
        ranks = generate_ranks(true, rank, size, options.start_port,
                               options.rxbuf_size);
    } else {
        ranks = generate_ranks(options.config_file, rank, options.start_port,
                               options.rxbuf_size);
    }

    xrt::device device{};
    if (options.hardware) {
        device = xrt::device(options.device_index);
    }

    std::unique_ptr<ACCL::ACCL> accl = initialize_accl(
        ranks, rank, !options.hardware, design, device, options.xclbin, 16,
        options.rxbuf_size, options.segment_size, options.rsfec);

    accl->set_timeout(1e6);

    MPI_Barrier(MPI_COMM_WORLD);
    test_vadd_put(*accl, device, options);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!options.hardware){
        test_reduce_stream(*accl, options);
        MPI_Barrier(MPI_COMM_WORLD);
        std::srand(42);
        for(int i=0; i<options.nruns; i++){
            unsigned char stream_id = std::abs(std::rand()) % 256;
            if(stream_id > 246) continue;
            test_loopback(*accl, options, stream_id);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        test_loopback_local_res(*accl, options);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
