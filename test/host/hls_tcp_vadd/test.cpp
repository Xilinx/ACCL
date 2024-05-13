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

// ACCL-specific includes
#include <accl.hpp>
#include <accl_network_utils.hpp>

// For initalizing the MPI environment
#include <mpi.h>

// For parsing CLI arguments
#include <tclap/CmdLine.h>

// For storing a list of the ACCL ranks/nodes (incl. IP address, port etc.)
#include <vector>

// For communicating with the FPGA from the host CPU
#include <xrt/xrt_device.h>

// Standard I/O
#include <iostream>

// HLS implementation of vector add kernel. Found in ACCL/kernels/plugins/vadd
#include "vadd_put.h"

// Used only during simulation; bus-functional model (BFM) of ACCL CCLO
#include "cclo_bfm.h"

struct options_t {
    unsigned int start_port;
    unsigned int rxbuf_size;
    unsigned int segment_size;
    unsigned int count;
    unsigned int device_index;
    bool hardware;
    bool rsfec;
    std::string xclbin;
    std::string config_file;
};

options_t parse_options(int argc, char *argv[]) {
    TCLAP::CmdLine cmd("Test HLS ACCL C++ driver. Performs local vector addition and sends the result to a neighbouring node");
    TCLAP::ValueArg<uint16_t> start_port_arg(
        "p", "start-port", "Start of range of ports", false, 5500, "positive integer"
    );
    cmd.add(start_port_arg);

    TCLAP::ValueArg<uint32_t> count_arg(
        "s", "count", "How many elements in the vector", false, 16, "positive integer"
    );
    cmd.add(count_arg);

    TCLAP::ValueArg<uint32_t> bufsize_arg(
        "b", "rxbuf-size", "How many KB per RX buffer", false, 1, "positive integer"
    );
    cmd.add(bufsize_arg);

    TCLAP::SwitchArg hardware_arg(
        "f", "hardware", "Enable hardware mode", cmd, false
    );

    TCLAP::ValueArg<std::string> xclbin_arg(
        "x", "xclbin", "xclbin file of ACCL driver if hardware mode is used", false, "accl.xclbin", "file"
    );
    cmd.add(xclbin_arg);
    
    TCLAP::ValueArg<uint16_t> device_index_arg(
        "i", "device-index", "device index of FPGA if hardware mode is used", false, 0, "positive integer"
    );
    cmd.add(device_index_arg);
    
    TCLAP::ValueArg<std::string> config_arg(
        "c", "config", "Config file containing IP mapping", false, "", "JSON file"
    );
    cmd.add(config_arg);
    
    TCLAP::SwitchArg rsfec_arg(
        "", "rsfec", "Enables RS-FEC in CMAC.", cmd, false
    );

    try {
        cmd.parse(argc, argv);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        exit(1);
    }

    options_t opts;
    opts.start_port = start_port_arg.getValue();
    opts.count = count_arg.getValue();
    opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
    opts.segment_size = opts.rxbuf_size;
    opts.hardware = hardware_arg.getValue();
    opts.xclbin = xclbin_arg.getValue();
    opts.device_index = device_index_arg.getValue();
    opts.config_file = config_arg.getValue();
    opts.rsfec = rsfec_arg.getValue();
    return opts;
}

void test_vadd_put(ACCL::ACCL &accl, xrt::device &device, options_t options, int current_rank, int world_size) {
    // Allocate float arrays for the HLS function to use
    float src[options.count], dst[options.count];
    for(int i = 0; i < options.count; i++){
        src[i] = 1.0 * (options.count * current_rank + i);
    }

    if (options.hardware) {
        // Instantiate vector-addition kernel from hardware
        xrt::kernel vadd_ip = xrt::kernel(
            device, 
            device.get_xclbin_uuid(), 
            "vadd_put:{vadd_0_0}",
            xrt::kernel::cu_access_mode::exclusive
        );

        // Allocated buffers for input and output data
        // Need to use XRT API because vector-addition kernel might use different HBM banks than ACCL
        auto src_bo = xrt::bo(device, sizeof(float) * options.count, vadd_ip.group_id(0));
        auto dst_bo = xrt::bo(device, sizeof(float) * options.count, vadd_ip.group_id(1));
        
        // Sync data, run kernel and wait for output
        src_bo.write(src);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        xrt::run run = vadd_ip(
            src_bo, dst_bo, options.count, 
            (current_rank + 1) % world_size, 
            accl.get_communicator_addr(),
            accl.get_arithmetic_config_addr({ACCL::dataType::float32, ACCL::dataType::float32})
        );
        run.wait(10000);
        
        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        dst_bo.read(dst);

    } else {
        // Initialize a CCLO BFM (simulation-only) and streams as needed
        hlslib::Stream<command_word> callreq, callack;
        hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
        std::vector<unsigned int> dest = {9};
        CCLO_BFM cclo(
            options.start_port, current_rank, world_size, 
            dest, callreq, callack, data_cclo2krnl, data_krnl2cclo
        );
        cclo.run();
        std::cout << "CCLO BFM started" << std::endl;

        // Wait for all nodes to initalize the BFM
        MPI_Barrier(MPI_COMM_WORLD);

        // Run the HLS function, using the global communicator
        vadd_put(
            src, dst, options.count,
            (current_rank + 1) % world_size,
            accl.get_communicator_addr(),
            accl.get_arithmetic_config_addr({ACCL::dataType::float32, ACCL::dataType::float32}),
            callreq, callack,
            data_krnl2cclo, data_cclo2krnl
        );

        // Stop the BFM
        cclo.stop();
    }

    // Check HLS function outputs
    unsigned int err_count = 0;
    for(int i=0; i < options.count; i++){
        float expected = 1.0 * (options.count*((current_rank + world_size - 1) % world_size) + i) + 1;
        if(dst[i] != expected){
            err_count++;
            std::cout << "Mismatch at [" << i << "]: got " << dst[i] << " vs expected " << expected << std::endl;
        }
    }

    std::cout << "RANK: " << current_rank << " - TEST FINISHED WITH " << err_count << " ERRORS!" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    // Initialize the MPI world, identify the id (current_rank) of the node
    MPI_Init(&argc, &argv);
    int current_rank, world_size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::cout << "current_rank: " << current_rank << " world_size: " << world_size << std::endl;

    // Parse CLI arguments
    options_t options = parse_options(argc, argv);

    // Generate a list of ACCL ranks (incl. IP, port, RX buffer size)
    std::vector<ACCL::rank_t> ranks;
    if (options.config_file == "") {
        ranks = accl_network_utils::generate_ranks(
            true, current_rank, world_size, options.start_port, options.rxbuf_size
        );
    } else {
        ranks = accl_network_utils::generate_ranks(
            options.config_file, current_rank, options.start_port, options.rxbuf_size
        );
    }

    // Initialize ACCL
    xrt::device device{};
    if (options.hardware) device = xrt::device(options.device_index);
    accl_network_utils::acclDesign design = accl_network_utils::acclDesign::TCP;
    std::unique_ptr<ACCL::ACCL> accl = accl_network_utils::initialize_accl(
        ranks, current_rank, !options.hardware, design, device, options.xclbin, 16,
        options.rxbuf_size, options.segment_size, options.rsfec
    );
    accl->set_timeout(1e6);

    // Wait until all ranks have finished setting-up and run test
    MPI_Barrier(MPI_COMM_WORLD);
    test_vadd_put(*accl, device, options, current_rank, world_size);
    
    // Finalize
    MPI_Finalize();
    return 0;
}
