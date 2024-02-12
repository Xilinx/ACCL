/*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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
*******************************************************************************/
#pragma once

#include <accl.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>
#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>
#include <xrt/xrt_kernel.h>

// TODO: Properly document functions in this header

namespace accl_network_utils {
enum class acclDesign { AXIS3x, TCP, UDP, CYT_TCP, CYT_RDMA };

// Error used for runtime network problems
class network_error : public std::runtime_error {
  // inherit constructor
  using std::runtime_error::runtime_error;
};

// Generate ranks based on JSON files with IPs
std::vector<ACCL::rank_t> generate_ranks(std::filesystem::path config_file,
                                         int local_rank,
                                         std::uint16_t start_port = 5500,
                                         unsigned int rxbuf_size = 1024);
// Generate ranks with IPs in private IP subnets
std::vector<ACCL::rank_t> generate_ranks(bool local, int local_rank,
                                         int world_size,
                                         std::uint16_t start_port = 5500,
                                         unsigned int rxbuf_size = 1024);

// Initialize accl and required network kernels
// If segsize == 0, the bufsize will be used as segment size instead
std::unique_ptr<ACCL::ACCL>
initialize_accl(const std::vector<ACCL::rank_t> &ranks, int local_rank,
                bool simulator, acclDesign design,
                xrt::device device = xrt::device(),
                std::filesystem::path xclbin = "", int nbufs = 16,
                ACCL::addr_t bufsize = 1024, ACCL::addr_t segsize = 0,
                bool rsfec = false);

// Configure the VNX kernel, this function is called by initialize_accl
void configure_vnx(vnx::CMAC &cmac, vnx::Networklayer &network_layer,
                   const std::vector<ACCL::rank_t> &ranks, int local_rank,
                   bool rsfec = false);

// Configure the TCP kernel, this function is called by initialize_accl
void configure_tcp(ACCL::BaseBuffer &tx_buf_network, ACCL::BaseBuffer &rx_buf_network, 
                   xrt::kernel &network_krnl, xrt::kernel &session_krnl,
                   const std::vector<ACCL::rank_t> &ranks, int local_rank);

// Get IPs from config file, this function is called by generate_ranks
std::vector<std::string> get_ips(std::filesystem::path config_file);
// Generate IPs in private subnet, this function is called by generate_ranks
std::vector<std::string> get_ips(bool local, int world_size);
} // namespace accl_network_utils
