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

#include <experimental/xrt_xclbin.h>

#include <unordered_map>
#include <vector>

namespace stream_scan {

/**
 * @brief Streaming Connectivity Scanning Class
 *
 */
class StreamScan {
private:
  struct XCLFSections {
    const ip_layout *ips_sec;
    const connectivity *con_sec;
    const mem_topology *mem_sec;

    XCLFSections(xrt::xclbin &xclbin_handle);
  };

  struct arg_type {
    uint32_t idx;
    std::string name;
    uint32_t mem_idx;
    uint32_t ip_idx;
  };

  struct mem_type {
    uint32_t idx;
    std::vector<uint32_t> ip_idx;
    std::vector<uint32_t> arg_idx;
  };

  struct ip_type {
    uint32_t idx;
    std::string name;
    std::vector<std::pair<uint32_t, ip_type *>> connections;
  };

private:
  xrt::xclbin xclbin_handle;
  XCLFSections *sections;
  std::unordered_map<uint32_t, ip_type> ips;
  std::unordered_map<uint32_t, mem_type> mems;
  std::unordered_map<uint32_t, arg_type> args;

  void map();

public:
  /**
   * @brief Represents existing streaming path between IP \a from to every IP in
   * \a to
   *
   */
  struct ip_map {
    std::string from;
    std::vector<std::string> to;
  };

  /**
   * @brief Empty constructor for StreamScan class
   *
   */
  StreamScan();

  /**
   * @brief Constructs StreamScan class from xclbin file path
   *
   * @param xclbin_path Path to the xclbin file to be set as current xclbin file
   * handle
   */
  StreamScan(std::string xclbin_path);

  /**
   * @brief Constructs StreamScan class from an xclbin handle
   *
   * @param xclbin_handle Handle to become the current xclbin file handle
   */
  StreamScan(xrt::xclbin &xclbin_handle);

  /**
   * @brief Set the current xclbin file handle from xclbin file path
   *
   * @param xclbin_path Path to the xclbin file to be set as current xclbin file
   * handle
   */
  void set_xclbin(std::string xclbin_path);

  /**
   * @brief Set the current xclbin file handle from another xclbin file handle
   *
   * @param xclbin_handle Handle to become the current xclbin file handle
   */
  void set_xclbin(xrt::xclbin &xclbin_handle);

  /**
   * @brief Returns the handle for the current xclbin file handle
   *
   * @return xrt::xclbin& Current xclbin file handle
   */
  xrt::xclbin &get_xclbin();

  /**
   * @brief Maps IP paths. Find any connection between \p start substring and \p
   * end substring by evaluating the stream mapping
   *
   * @param start If \p arg is true, represents an argument substring,
   * otherwise, represents an IP substring
   * @param end Represents the IP substring of the path end
   * @param arg If set to true, \p start is an argument substring, otherwise, it
   * is an IP substring, default is false
   * @param max_steps maxing number of IP hoping, default is -1, which set max
   * hops to infinity
   * @return std::vector<ip_map> Array with resulting IP to IP connections
   */
  std::vector<ip_map> mapIPs(std::string start, std::string end,
                             bool arg = false, int32_t max_steps = -1);

  /**
   * @brief Search for IPs that match the given description
   *
   * @param desc Substring that describe the IP or Argument description
   * @param arg If set to true, \p desc is an argument substring, otherwise, it
   * is an IP substring, default is false
   * @return std::vector<std::string> Array with resulting IPs
   */
  std::vector<std::string> findIPs(std::string desc, bool arg = false);

  /**
   * @brief Gets the id of banks the IP connects to. This returns the banks in
   * general, not in IP argument order
   *
   * @param ip_name Name of the IP to get connections
   * @param args_desc If set, the resulting array will only display banks for
   * the given args, default is empty array
   * @param inc_stream If set, results will include stream_connect args banks,
   * otherwise won't, default is false
   * @return std::vector<uint32_t> List with the banks IP arguments connect to
   */
  std::vector<uint32_t> getIPbanks(std::string ip_name,
                                   std::vector<std::string> args_desc = {},
                                   bool inc_stream = false);

  /**
   * @brief Dumps information about the Memory Section of the xclbin xclf
   * section
   *
   */
  void dump_mem_section();

  /**
   * @brief Dumps information about the IP Layout Section of the xclbin xclf
   * section
   *
   */
  void dump_ips_section();

  /**
   * @brief Dumps information about the Connectivity Section of the xclbin xclf
   * section
   *
   */
  void dump_con_section();
};
} // namespace stream_scan
