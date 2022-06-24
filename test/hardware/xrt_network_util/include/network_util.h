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
#  Based on the xup vitis network example
#  (https://github.com/Xilinx/xup_vitis_network_example/tree/host_xrt)
#  Copyright (C) FPGA-FAIS at Jagiellonian University Cracow
#  SPDX-License-Identifier: BSD-3-Clause
#
*******************************************************************************/

#pragma once
#include <arpa/inet.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <experimental/xrt_ip.h>
#include <ostream>

namespace ACCL_NETWORK_UTIL {
const std::size_t socket_count = 16;

class CMAC {
public:
  /* Offsets taken from
   * test/hardware/xup_vitis_network_example/ethernet/template.xml */
  const uint32_t stat_tx_status = 0x0200;
  const uint32_t stat_rx_status = 0x0204;
  const uint32_t stat_pm_tick = 0x02B0;
  const uint32_t stat_rx_total_packets = 0x0608;

  struct cmac_status {
    bool cmac_link;
    bool rx_status;
    bool rx_aligned;
    bool rx_misaligned;
    bool rx_aligned_err;
    bool rx_hi_ber;
    bool rx_remote_fault;
    bool rx_local_fault;
    bool rx_got_signal_os;
    bool tx_local_fault;
  };

  CMAC(xrt::ip &cmac) : cmac(cmac) {}

  CMAC(xrt::ip &&cmac) : cmac(cmac) {}

  ~CMAC() {}

  inline cmac_status cmac_link_status() {
    uint32_t rx_status, tx_status;
    cmac_status status;
    // The first time these registers are not populated properly,
    // read them twice to get real value
    for (std::size_t i = 0; i < 2; ++i) {
      rx_status = cmac.read_register(stat_rx_status);
      tx_status = cmac.read_register(stat_tx_status);
    }

    status.cmac_link = shifted_word(rx_status, 0);
    status.rx_status = shifted_word(rx_status, 0);
    status.rx_aligned = shifted_word(rx_status, 1);
    status.rx_misaligned = shifted_word(rx_status, 2);
    status.rx_aligned_err = shifted_word(rx_status, 3);
    status.rx_hi_ber = shifted_word(rx_status, 4);
    status.rx_remote_fault = shifted_word(rx_status, 5);
    status.rx_local_fault = shifted_word(rx_status, 6);
    status.rx_got_signal_os = shifted_word(rx_status, 14);
    status.tx_local_fault = shifted_word(tx_status, 0);

    return status;
  }

  inline void cmac_copy_stats() { cmac.write_register(stat_pm_tick, 1); }

  inline void cmac_get(bool update_reg) {
    // TODO
  }

private:
  xrt::ip cmac;

  template <typename T>
  inline T shifted_word(T value, std::size_t index, std::size_t width = 1) {
    return (value >> index) & ((1 << width) - 1);
  }
};

std::ostream &operator<<(std::ostream &os, const CMAC::cmac_status &status) {
  return os << "{cmac_link: " << status.cmac_link
            << ", rx_status: " << status.rx_status
            << ", rx_aligned: " << status.rx_aligned
            << ", rx_misaligned: " << status.rx_misaligned
            << ", rx_aligned_err: " << status.rx_aligned_err
            << ", rx_hi_ber: " << status.rx_hi_ber
            << "rx_remote_fault: " << status.rx_remote_fault
            << "rx_local_fault: " << status.rx_local_fault
            << "rx_got_signal_os: " << status.rx_got_signal_os
            << "tx_local_fault: " << status.tx_local_fault << "}";
}

class NetworkLayer {
public:
  /* Offsets taken from
   * test/hardware/xup_vitis_network_example/NetLayers/kernel.xml */
  const uint32_t mac_address_lsb = 0x0010;
  const uint32_t mac_address_msb = 0x0014;
  const uint32_t ip_address = 0x0018;
  const uint32_t gateway = 0x001c;
  const uint32_t ip_mask = 0x0020;

  const uint32_t arp_discovery_offset = 0x3010;

  const uint32_t eth_in_packets = 0x1010;
  const uint32_t eth_out_packets = 0x10B8;

  const uint32_t udp_theirIP_offset = 0x2010;
  const uint32_t udp_theirPort_offset = 0x2090;
  const uint32_t udp_myPort_offset = 0x2110;
  const uint32_t udp_valid_offset = 0x2190;
  const uint32_t udp_number_sockets_offset = 0x2210;

  struct socket_t {
    uint32_t theirIP;
    uint16_t theirPort;
    uint16_t myPort;
    bool valid;
  };

  NetworkLayer(xrt::ip &network_layer) : network_layer(network_layer) {}
  NetworkLayer(xrt::ip &&network_layer) : network_layer(network_layer) {}

  ~NetworkLayer() {}

  void populate_socket_table() {
    uint32_t udp_number_sockets_hw =
        network_layer.read_register(udp_number_sockets_offset);
    if (udp_number_sockets_hw < socket_count) {
      throw std::runtime_error(
          "Socket list length (" + std::to_string(socket_count) +
          ") is bigger than the number of sockets in hardware (" +
          std::to_string(udp_number_sockets_hw) + ")");
    }

    for (std::size_t i = 0; i < socket_count; ++i) {
      network_layer.write_register(udp_theirIP_offset + i * 8,
                                   sockets[i].theirIP);
      network_layer.write_register(udp_theirPort_offset + i * 8,
                                   sockets[i].theirPort);
      network_layer.write_register(udp_myPort_offset + i * 8,
                                   sockets[i].myPort);
      network_layer.write_register(udp_valid_offset + i * 8, sockets[i].valid);
    }
  }

  void arp_discovery() {
    network_layer.write_register(arp_discovery_offset, 0);
    network_layer.write_register(arp_discovery_offset, 1);
    network_layer.write_register(arp_discovery_offset, 0);
  }

  void update_ip_address(std::string ip) {
    uint32_t ip_encoded = ip_encode(ip);
    network_layer.write_register(ip_address, ip_encoded);
    network_layer.write_register(gateway, (ip_encoded & 0xFFFFFF00) + 1);

    uint64_t mac =
        ((uint64_t)network_layer.read_register(mac_address_msb) << 32) +
        network_layer.read_register(mac_address_lsb);
    mac = (mac & 0xFFFFFFFFF00) + (ip_encoded & 0xFF);
    network_layer.write_register(mac_address_msb,
                                 static_cast<uint32_t>(mac >> 32));
    network_layer.write_register(mac_address_lsb,
                                 static_cast<uint32_t>(mac & 0xFFFFFFFF));
  }

  void set_socket(std::size_t index, std::string &theirIP, uint16_t theirPort,
                  uint16_t ourPort, bool valid) {
    socket_t sock = {ip_encode(theirIP), theirPort, ourPort, valid};
    sockets[index] = sock;
  }

private:
  xrt::ip network_layer;
  std::array<socket_t, socket_count> sockets{};

  inline void swap_endianness(uint32_t *ip) {
    uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
    *ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
          (ip_bytes[0] << 24);
  }

  uint32_t ip_encode(std::string ip) {
    struct sockaddr_in sa;
    inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
    swap_endianness(&sa.sin_addr.s_addr);
    return sa.sin_addr.s_addr;
  }

  std::string ip_decode(uint32_t ip) {
    char buffer[INET_ADDRSTRLEN];
    struct in_addr sa;
    sa.s_addr = ip;
    swap_endianness(&sa.s_addr);
    inet_ntop(AF_INET, &sa, buffer, INET_ADDRSTRLEN);
    return std::string(buffer, INET_ADDRSTRLEN);
  }
};
} // namespace ACCL_NETWORK_UTIL
