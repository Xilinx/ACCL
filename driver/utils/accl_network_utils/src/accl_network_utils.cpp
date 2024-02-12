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
#include <fstream>
#include <json/json.h>

#ifdef ACCL_NETWORK_UTILS_MPI
#include <mpi.h>
#else
#include <chrono>
#include <thread>
#endif // ACCL_NETWORK_UTILS_MPI

#include "accl_network_utils.hpp"

using namespace ACCL;
namespace fs = std::filesystem;

namespace {

/**
 * Insert barrier when using MPI, otherwise sleep for 3 seconds.
 *
 */
inline void pause_barrier() {
#ifdef ACCL_NETWORK_UTILS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#else
  std::this_thread::sleep_for(std::chrono::seconds(3));
#endif // ACCL_NETWORK_UTILS_MPI
}

/**
 * Insert barrier when using MPI.
 *
 */
inline void barrier() {
#ifdef ACCL_NETWORK_UTILS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif // ACCL_NETWORK_UTILS_MPI
}

/**
 * Log message when debugging is enabled.
 *
 */
inline void log_debug(std::string debug) {
#ifdef ACCL_NETWORK_UTILS_DEBUG
  std::cerr << debug << std::endl;
#endif // ACCL_NETWORK_UTILS_DEBUG
}

/**
 * Check if ARP table contains all ranks and doesn't contain duplicates.
 *
 */
bool check_arp(vnx::Networklayer &network_layer,
               const std::vector<rank_t> &ranks, int local_rank,
               int world_size) {
  std::map<unsigned, bool> ranks_checked;
  for (unsigned i = 0; i < static_cast<unsigned>(world_size); ++i) {
    ranks_checked[i] = false;
  }

  bool sanity_check = true;
  const std::map<int, std::pair<std::string, std::string>> arp =
      network_layer.read_arp_table(world_size);

  std::ostringstream ss_arp;
  ss_arp << "ARP table:";

  for (const std::pair<const int, std::pair<std::string, std::string>> &elem :
       arp) {
    const unsigned index = elem.first;
    const std::pair<std::string, std::string> &entry = elem.second;
    const std::string &mac = entry.first;
    const std::string &ip = entry.second;
    ss_arp << "\n(" << index << ") " << mac << ": " << ip;

    for (unsigned i = 0; i < static_cast<unsigned>(world_size); ++i) {
      if (ranks[i].ip == ip) {
        if (ranks_checked[i]) {
          std::cout << "Double entry for " << ip << " in arp table!"
                    << std::endl;
          sanity_check = false;
        } else {
          ranks_checked[i] = true;
        }
      }
    }
  }

  log_debug(ss_arp.str());

  if (!sanity_check) {
    return false;
  }

  unsigned hosts = 0;
  for (unsigned i = 0; i < static_cast<unsigned>(world_size); ++i) {
    if (ranks_checked[i]) {
      hosts += 1;
    }
  }
  if (hosts < static_cast<unsigned>(world_size) - 1) {
    std::cout << "Found only " << hosts << " hosts out of " << world_size - 1
              << "!" << std::endl;
    return false;
  }

  return true;
}
} // namespace

namespace accl_network_utils {
void configure_vnx(vnx::CMAC &cmac, vnx::Networklayer &network_layer,
                   const std::vector<rank_t> &ranks, int local_rank,
                   bool rsfec) {
  if (ranks.size() > vnx::max_sockets_size) {
    throw std::runtime_error("Too many ranks. VNX supports up to " +
                             std::to_string(vnx::max_sockets_size) +
                             " sockets.");
  }

  if (cmac.get_rs_fec() != rsfec) {
    std::cout << "Turning RS-FEC " << (rsfec ? "on" : "off") << "..."
              << std::endl;
    cmac.set_rs_fec(rsfec);
  }

  std::cout << "Testing UDP link status: ";

  const auto link_status = cmac.link_status();

  if (link_status.at("rx_status")) {
    std::cout << "Link successful!" << std::endl;
  }

  std::ostringstream ss;

  ss << "Link interface 1 : {";
  for (const auto &elem : link_status) {
    ss << elem.first << ": " << elem.second << ", ";
  }
  ss << "}" << std::endl;
  log_debug(ss.str());

  if (!link_status.at("rx_status")) {
    throw network_error("No link found.");
  }

  barrier();

  std::cout << "Populating socket table..." << std::endl;

  network_layer.update_ip_address(ranks[local_rank].ip);
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (i == static_cast<size_t>(local_rank)) {
      continue;
    }

    network_layer.configure_socket(i, ranks[i].ip, ranks[i].port,
                                   ranks[local_rank].port, true);
  }

  network_layer.populate_socket_table();

  std::cout << "Starting ARP discovery..." << std::endl;
  pause_barrier();
  network_layer.arp_discovery();
  std::cout << "Finishing ARP discovery..." << std::endl;
  pause_barrier();
  network_layer.arp_discovery();
  std::cout << "ARP discovery finished!" << std::endl;

  if (!check_arp(network_layer, ranks, local_rank, ranks.size())) {
    throw network_error("Problem in ARP table.");
  }
}

void configure_tcp(FPGABuffer<int8_t> &tx_buf_network, FPGABuffer<int8_t> &rx_buf_network,
                   xrt::kernel &network_krnl, xrt::kernel &session_krnl,
                   const std::vector<rank_t> &ranks, int local_rank) {
  tx_buf_network.sync_to_device();
  rx_buf_network.sync_to_device();

  uint local_fpga_ip = ip_encode(ranks[local_rank].ip);
  log_debug("rank: " + std::to_string(local_rank) +
            " FPGA IP: " + std::to_string(local_fpga_ip));

  network_krnl(local_fpga_ip, static_cast<uint32_t>(local_rank), local_fpga_ip,
               *(tx_buf_network.bo()), *(rx_buf_network.bo()));

  uint32_t ip_reg = network_krnl.read_register(0x010);
  uint32_t board_reg = network_krnl.read_register(0x018);
  std::ostringstream ss;
  ss << std::hex << "ip_reg: " << ip_reg << " board_reg IP: " << board_reg
     << std::dec << std::endl;
  log_debug(ss.str());

  //set up sessions for ranks
  for(size_t i = 0; i < ranks.size(); ++i){
    bool success;
    if (i == static_cast<size_t>(local_rank)) {
      continue;
    }
    session_krnl(ranks[i].ip, ranks[i].port, false, 	
                  &(ranks[i].session_id), &success);
    if(!success){
      throw std::runtime_error("Failed to establish session for IP:"+
                                ranks[i].ip+
                                " port: "+
                                std::to_string(ranks[i].port));
    }
    std::ostringstream ss;
    ss << "Established session ID: " << ranks[i].session_id << std::endl;
    log_debug(ss.str());
  }

}

std::vector<std::string> get_ips(fs::path config_file) {
  std::vector<std::string> ips{};
  Json::Value config;
  std::ifstream config_file_stream(config_file);
  config_file_stream >> config;
  Json::Value ip_config = config["ips"];
  int size = ip_config.size();
  if (size < 1) {
    throw std::runtime_error("IPs not specified in config file");
  }

  for (int i = 0; i < size; ++i) {
    ips.push_back(ip_config[i].asString());
  }

  return ips;
}

std::vector<std::string> get_ips(bool local, int world_size) {
  std::vector<std::string> ips{};
  for (int i = 0; i < world_size; ++i) {
    if (local) {
      ips.emplace_back("127.0.0.1");
    } else {
      ips.emplace_back("10.10.10." + std::to_string(i + 1));
    }
  }
  return ips;
}

std::vector<rank_t> generate_ranks(fs::path config_file, int local_rank,
                                   std::uint16_t start_port,
                                   unsigned int rxbuf_size) {
  std::vector<rank_t> ranks{};
  std::vector<std::string> ips = get_ips(config_file);

  for (int i = 0; i < static_cast<int>(ips.size()); ++i) {
    rank_t new_rank = {ips[i], start_port + i, i, rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  return ranks;
}

std::vector<rank_t> generate_ranks(bool local, int local_rank, int world_size,
                                   std::uint16_t start_port,
                                   unsigned int rxbuf_size) {
  std::vector<rank_t> ranks{};
  std::vector<std::string> ips = get_ips(local, world_size);
  for (int i = 0; i < static_cast<int>(ips.size()); ++i) {
    rank_t new_rank = {ips[i], start_port + i, i, rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  return ranks;
}

std::unique_ptr<ACCL::ACCL>
initialize_accl(const std::vector<rank_t> &ranks, int local_rank,
                bool simulator, acclDesign design, xrt::device device,
                fs::path xclbin, int nbufs, addr_t bufsize, addr_t segsize,
                bool rsfec) {
  std::size_t world_size = ranks.size();
  std::unique_ptr<ACCL::ACCL> accl;

  if (segsize == 0) {
    segsize = bufsize;
  }

  if (simulator) {
    accl = std::make_unique<ACCL::ACCL>(ranks[0].port, local_rank);
  } else {
    int devicemem;
    std::vector<int> rxbufmem;
    int networkmem;

    std::string cclo_id;
    if (design == acclDesign::AXIS3x) {
      cclo_id = std::to_string(local_rank);
    } else {
      cclo_id = "0";
    }

    auto xclbin_uuid = device.load_xclbin(xclbin.string());
    auto cclo_ip = xrt::ip(device, xclbin_uuid,
                           "ccl_offload:{ccl_offload_" + cclo_id + "}");
    auto hostctrl_ip = xrt::kernel(device, xclbin_uuid,
                                   "hostctrl:{hostctrl_" + cclo_id + "_0}",
                                   xrt::kernel::cu_access_mode::exclusive);

    if (design == acclDesign::AXIS3x) {
      devicemem = local_rank * 6;
      rxbufmem = {local_rank * 6 + 1};
      networkmem = local_rank * 6 + 2;
    } else {
      devicemem = 0;
      rxbufmem = {1};
      networkmem = 6;
    }

    if (design == acclDesign::UDP) {
      auto cmac = vnx::CMAC(xrt::ip(device, xclbin_uuid, "cmac_0:{cmac_0}"));
      auto network_layer = vnx::Networklayer(
          xrt::ip(device, xclbin_uuid, "networklayer:{networklayer_0}"));

      configure_vnx(cmac, network_layer, ranks, local_rank, rsfec);
    } else if (design == acclDesign::TCP) {
      // Tx and Rx buffers will not be cleaned up properly and leak memory.
      // They need to live at least as long as ACCL so for now this is the best
      // we can do without requiring the users to allocate the buffers manually.
      auto tx_buf_network = new FPGABuffer<int8_t>(
          64 * 1024 * 1024, dataType::int8, device, networkmem);
      auto rx_buf_network = new FPGABuffer<int8_t>(
          64 * 1024 * 1024, dataType::int8, device, networkmem);
      auto network_krnl =
          xrt::kernel(device, xclbin_uuid, "network_krnl:{network_krnl_0}",
                      xrt::kernel::cu_access_mode::exclusive);
      auto session_krnl =
          xrt::kernel(device, xclbin_uuid, "tcp_session_handler:{session_handler_0}",
                      xrt::kernel::cu_access_mode::exclusive);
      configure_tcp(*tx_buf_network, *rx_buf_network, network_krnl, session_krnl,
                     ranks, local_rank);
    }

    accl = std::make_unique<ACCL::ACCL>(device, cclo_ip, hostctrl_ip, devicemem, rxbufmem);
  }
  accl.get()->initialize(ranks, local_rank,	nbufs, bufsize, segsize);
  return accl;
}
} // namespace accl_network_utils
