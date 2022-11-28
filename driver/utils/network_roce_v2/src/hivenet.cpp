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

#include "roce/hivenet.hpp"
#include <cmath>
#include <iostream>
#include <sstream>

namespace roce {
uint32_t encode_ip_address(const std::string decoded_ip) {
  std::vector<std::string> l_ipAddrVec;
  std::stringstream l_str(decoded_ip);
  std::string l_ipAddrStr;
  if (std::getline(l_str, l_ipAddrStr, '.').fail()) {
    throw std::runtime_error("IP address is ill-formed.");
    return 0;
  } else {
    l_ipAddrVec.push_back(l_ipAddrStr);
  }
  while (std::getline(l_str, l_ipAddrStr, '.')) {
    l_ipAddrVec.push_back(l_ipAddrStr);
  }
  if (l_ipAddrVec.size() != 4) {
    throw std::runtime_error("IP address is ill-formed.");
    return 0;
  }
  uint32_t l_ipAddr = 0;
  for (auto i = 0; i < 4; ++i) {
    l_ipAddr = l_ipAddr << 8;
    uint32_t l_val = std::stoi(l_ipAddrVec[i]);
    if (l_val > 255) {
      std::string l_errStr = l_ipAddrVec[i] + " should be less than 255.";
      throw std::runtime_error(l_errStr);
      return 0;
    }
    l_ipAddr += l_val;
  }
  return l_ipAddr;
}

std::string decode_ip_address(const uint32_t encoded_ip) {
  std::string l_ipStr;
  for (auto i = 0; i < 4; ++i) {
    uint32_t l_ipAddr = encoded_ip;
    l_ipAddr = l_ipAddr >> (4 - i - 1) * 8;
    uint8_t l_digit = l_ipAddr & 0xff;
    l_ipStr = l_ipStr + std::to_string(l_digit);
    if (i != 3) {
      l_ipStr += ".";
    }
  }
  return l_ipStr;
}

uint16_t Hivenet::get_local_id() {
  uint16_t localID = hivenet.read_register(localID_off);
  return localID;
}

void Hivenet::init_hivenet() {
  reset_arp_table();
  set_timeout(65535);
  set_retransmissions(3);
  set_udp_port(4971);
}

void Hivenet::set_timeout(const uint64_t timeout) {
  hivenet.write_register(timeout_off, timeout);
  hivenet.write_register(timeout_off + 4, timeout >> 32);
}

void Hivenet::set_ip_subnet(const uint32_t ip_subnet) {
  hivenet.write_register(IPsubnet_off, ip_subnet);
}

uint32_t Hivenet::set_ip_subnet(const std::string ip_subnet) {
  uint32_t encoded_ip_subnet = encode_ip_address(ip_subnet);
  set_ip_subnet(encoded_ip_subnet);
  return encoded_ip_subnet;
}

uint32_t Hivenet::get_ip_subnet_encoded() {
  uint32_t ip_subnet = hivenet.read_register(IPsubnet_off);
  return ip_subnet;
}

std::string Hivenet::get_ip_subnet() {
  uint32_t ip_subnet = hivenet.read_register(IPsubnet_off);
  return decode_ip_address(get_ip_subnet_encoded());
}

void Hivenet::set_mac_subnet(const uint64_t mac_subnet) {
  hivenet.write_register(MACSubnet_off, mac_subnet);
  hivenet.write_register(MACSubnet_off + 4, mac_subnet >> 32);
}

uint64_t Hivenet::get_mac_subnet() {
  uint64_t mac_subnet = hivenet.read_register(MACSubnet_off + 4);
  mac_subnet = mac_subnet << 32;
  mac_subnet += hivenet.read_register(MACSubnet_off);
  mac_subnet &= 0xffffffffffffe000;
  return mac_subnet;
}

void Hivenet::set_retransmissions(const uint16_t retransmissions) {
  hivenet.write_register(retransmissions_off, retransmissions);
}
uint16_t Hivenet::get_retransmissions() {
  uint16_t retransmissions = hivenet.read_register(retransmissions_off);
  return retransmissions;
}
void Hivenet::set_udp_port(const uint16_t udp_port) {
  hivenet.write_register(UDPPort_off, udp_port);
}

void Hivenet::reset_arp_table() {
  hivenet.write_register(resetARP_off, 1);
}

uint32_t Hivenet::get_ip_address_encoded() {
  uint16_t id = hivenet.read_register(localID_off);
  uint32_t subnet = hivenet.read_register(IPsubnet_off);
  uint32_t subnet_little =
      ((subnet >> 24) & 0xff) | ((subnet << 8) & 0xff0000) |
      ((subnet >> 8) & 0xff00) | ((subnet << 24) & 0xff000000);
  uint32_t ip_little = subnet_little + id;
  uint32_t ip_big = ((ip_little >> 24) & 0xff) | ((ip_little << 8) & 0xff0000) |
                    ((ip_little >> 8) & 0xff00) |
                    ((ip_little << 24) & 0xff000000);
  return ip_big;
}

std::string Hivenet::get_ip_address() {
  return decode_ip_address(get_ip_address_encoded());
}

uint64_t Hivenet::get_mac_address() {
  uint64_t macSubAddress = hivenet.read_register(MACSubnet_off + 4);
  macSubAddress = macSubAddress << 32;
  macSubAddress += hivenet.read_register(MACSubnet_off);
  uint16_t id = hivenet.read_register(localID_off);
  uint64_t macAddress = macSubAddress;
  macAddress &= 0xffffffffffffe000;
  macAddress = macAddress + id;
  return macAddress;
}
uint16_t Hivenet::get_udp_port() {
  uint16_t udp_port = hivenet.read_register(UDPPort_off);
  return udp_port;
}

uint32_t Hivenet::get_buf_size() { return hivenet.read_register(buffSize_off); }
uint64_t Hivenet::get_timeout() {
  uint64_t timeout = hivenet.read_register(timeout_off + 4);
  timeout = timeout << 32;
  timeout += hivenet.read_register(timeout_off);
  return timeout;
}
} // namespace roce
