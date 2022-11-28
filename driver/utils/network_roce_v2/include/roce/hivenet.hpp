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

#include <cstddef>
#include <cstdint>
#include <experimental/xrt_ip.h>
#include <map>
#include <string>
#include <vector>

namespace roce {
// Register offsets
constexpr std::size_t localID_off = 0x0010;
constexpr std::size_t timeout_off = 0x0018;
constexpr std::size_t IPsubnet_off = 0x0024;
constexpr std::size_t MACSubnet_off = 0x002C;
constexpr std::size_t retransmissions_off = 0x0038;
constexpr std::size_t UDPPort_off = 0x0040;
constexpr std::size_t resetARP_off = 0x0048;
constexpr std::size_t buffSize_off = 0x0100;

uint32_t encode_ip_address(const std::string decoded_ip);
std::string decode_ip_address(const uint32_t encoded_ip);

class Hivenet {
public:
  Hivenet(xrt::ip &hivenet, const uint32_t local_id) : hivenet(hivenet), local_id(local_id) {
    init_hivenet();
  }
  Hivenet(xrt::ip &&hivenet, const uint32_t local_id) : hivenet(hivenet), local_id(local_id) {
    init_hivenet();
  }

  /**
  * @brief setting the timeout parameter.
  *
  * @param timeout Timeout in nanoseconds for each packet in case of packet loss in network, default value 65535 ns.
  *
  */
  void set_timeout(const uint64_t timeout);

  /**
  * @brief setting the ipSubnet, the IP subnet should be the same for all HiveNet IPs.
  *
  * @param ip_subnet The encoded IP subnet the HiveNets are working in.
  *
  */
  void set_ip_subnet(const uint32_t ip_subnet);
  /**
  * @brief setting the ipSubnet, the IP subnet should be the same for all HiveNet IPs.
  *
  * @param ip_subnet The decoded IP subnet the HiveNets are working in.
  *
  */
  uint32_t set_ip_subnet(const std::string ip_subnet);

  /**
  * @brief setting the macSubnet, the MAC subnet should be the same for all HiveNet IPs.
  *
  * @param macSubnet The MAC subnet the HiveNets are working in.
  *
  */
  void set_mac_subnet(const uint64_t mac_subnet);

  /**
  * @brief setting the retransmissions.
  *
  * @param retransmissions number of retransmissions in case of packet loss in network, default value 3.
  *
  */
  void set_retransmissions(const uint16_t retransmissions);

  /**
  * @brief setting the udp port, the UDP port should be the same on all HiveNet IPs.
  *
  * @param udp_port the UDP port number, default value 4971.
  *
  */
  void set_udp_port(const uint16_t udp_port);

  /**
  * @brief resetting ARP table.
  */
  void reset_arp_table();

  /**
  * @brief read timeout setting.
  */
  uint64_t get_timeout();

  /**
  * @brief getting encoded IPSubnet.
  */
  uint32_t get_ip_subnet_encoded();

  /**
  * @brief getting decoded IPSubnet.
  */
  std::string get_ip_subnet();

  /**
  * @brief getting localID.
  */
  uint16_t get_local_id();

  /**
  * @brief getting MACSubnet.
  */
  uint64_t get_mac_subnet();

  /**
  * @brief getting number of retransmissions.
  */
  uint16_t get_retransmissions();

  /**
  * @brief getting UDPPort number.
  */
  uint16_t get_udp_port();

  /**
  * @brief getting encoded IPAddress, each HiveNet IP has an IP address calculated from IPSubnet and localID.
  */
  uint32_t get_ip_address_encoded();

  /**
  * @brief getting encoded IPAddress, each HiveNet IP has an IP address calculated from IPSubnet and localID.
  */
  std::string get_ip_address();

  /**
  * @brief getting MACAddress, each HiveNet IP has an Mac address calculated from MACsubnet and localID.
  */
  uint64_t get_mac_address();

  /**
  * @brief return number of  packets in the HiveNet buffer waiting for their acknowledgment.
  */
  uint32_t get_buf_size();

private:
  xrt::ip hivenet;
  const uint32_t local_id;

  void init_hivenet();
};
} // namespace roce
