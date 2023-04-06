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

#include <algorithm>
#include <iostream>
#include <queue>

#include "stream_scan.hpp"

namespace {
std::string mem2str(int mem_type) {
  switch (mem_type) {
  case 0:
    return "MEM_DDR3";
  case 1:
    return "MEM_DDR4";
  case 2:
    return "MEM_DRAM";
  case 3:
    return "MEM_STREAMING";
  case 4:
    return "MEM_PREALLOCATED_GLOB";
  case 5:
    return "MEM_ARE";
  case 6:
    return "MEM_HBM";
  case 7:
    return "MEM_BRAM";
  case 8:
    return "MEM_URAM";
  case 9:
    return "MEM_STREAMING_CONNECTION";
  case 10:
    return "MEM_HOST";
  }
  return "";
}

/**
 * Log message when debugging is enabled.
 *
 */
inline void log_debug(std::string debug) {
#ifdef STREAM_SCAN_DEBUG
  std::cerr << debug << std::endl;
#endif // STREAM_SCAN_DEBUG
}
} // namespace

namespace stream_scan {

StreamScan::XCLFSections::XCLFSections(xrt::xclbin &xclbin_handle) {
  ips_sec = xclbin_handle.get_axlf_section<const ip_layout *>(
      axlf_section_kind::IP_LAYOUT);
  con_sec = xclbin_handle.get_axlf_section<const connectivity *>(
      axlf_section_kind::CONNECTIVITY);
  mem_sec = xclbin_handle.get_axlf_section<const mem_topology *>(
      axlf_section_kind::MEM_TOPOLOGY);

  log_debug("Xclbin file sections ready.");
}

void StreamScan::map() {
  // Create ips map
  const ip_layout *ip_s = sections->ips_sec;
  for (uint32_t i = 0; i < ip_s->m_count; i++) {
    if (ip_s->m_ip_data[i].m_type == IP_TYPE::IP_KERNEL) {
      ips[i] = {
          .idx = i,
          .name = std::string((char *)ip_s->m_ip_data[i].m_name),
      };
    }
  }

  // Create mem map
  const mem_topology *mem_s = sections->mem_sec;
  for (uint32_t i = 0; i < mem_s->m_count; i++) {
    if (mem_s->m_mem_data[i].m_type == MEM_TYPE::MEM_STREAMING_CONNECTION) {
      mems[i] = {
          .idx = i,
      };
    }
  }

  // Link both maps
  const connectivity *con_s = sections->con_sec;
  for (uint32_t i = 0; i < con_s->m_count; i++) {
    uint32_t ip_id = con_s->m_connection[i].m_ip_layout_index;
    uint32_t mem_id = con_s->m_connection[i].mem_data_index;
    int32_t arg_id = con_s->m_connection[i].arg_index;

    xrt::xclbin::ip ip_rel = xclbin_handle.get_ip(ips[ip_id].name);

    args[i] = {
        .idx = i,
        .name = ip_rel.get_arg(arg_id).get_name(),
        .mem_idx = mem_id,
        .ip_idx = ip_id,
    };

    if (mems.find(mem_id) != mems.end()) {
      mems[mem_id].ip_idx.push_back(ip_id);
      mems[mem_id].arg_idx.push_back(i);
    }
  }

  for (auto &mem : mems) {
    ips[mem.second.ip_idx[0]].connections.push_back(
        {mem.second.arg_idx[1], &ips[mem.second.ip_idx[1]]});
    ips[mem.second.ip_idx[1]].connections.push_back(
        {mem.second.arg_idx[0], &ips[mem.second.ip_idx[0]]});
  }

  log_debug("IP streaming connections mapped.");
}

StreamScan::StreamScan() : sections(nullptr) {}

StreamScan::StreamScan(std::string xclbin_path) {
  xclbin_handle = xrt::xclbin(xclbin_path);
  sections = new XCLFSections(xclbin_handle);
  map();
}

StreamScan::StreamScan(xrt::xclbin &xclbin_handle)
    : xclbin_handle(xclbin_handle) {
  sections = new XCLFSections(xclbin_handle);
  map();
}

void StreamScan::set_xclbin(std::string xclbin_path) {
  xclbin_handle = xrt::xclbin(xclbin_path);
  sections = new XCLFSections(xclbin_handle);
  map();
}

void StreamScan::set_xclbin(xrt::xclbin &xclbin_handle) {
  xclbin_handle = xclbin_handle;
  sections = new XCLFSections(xclbin_handle);
  map();
}

xrt::xclbin &StreamScan::get_xclbin() { return xclbin_handle; }

std::vector<StreamScan::ip_map> StreamScan::mapIPs(std::string start,
                                                   std::string end, bool arg,
                                                   int32_t max_steps) {
  std::vector<int32_t> helper;
  std::vector<StreamScan::ip_map> ip_mapping;

  if (sections == nullptr) {
    log_debug("Sections were not initialized");
    return ip_mapping;
  }

  if (!arg) {
    for (auto &ip : ips) {
      if (ip.second.name.find(start) != std::string::npos) {
        if (ip.second.name.find(":") != std::string::npos) {
          ip_mapping.push_back({.from = ip.second.name});
          helper.push_back(ip.first);
        }
      }
    }
  } else {
    for (auto &arg : args) {
      if (arg.second.name.find(start) != std::string::npos) {
        ip_mapping.push_back({.from = ips[arg.second.ip_idx].name});
        helper.push_back(arg.second.ip_idx);
      }
    }
  }

  for (uint32_t i = 0; i < ip_mapping.size(); i++) {
    std::vector<bool> visited(ips.size(), false);
    visited[helper[i]] = true;

    std::queue<ip_type *> ip_queue;
    std::queue<int32_t> ip_step;
    ip_queue.push(&ips[helper[i]]);
    ip_step.push(0);

    while (!ip_queue.empty()) {
      ip_type *curr = ip_queue.front();
      ip_queue.pop();
      int32_t step = ip_step.front();
      ip_step.pop();

      if (max_steps != -1 && step >= max_steps)
        continue;

      for (auto entry : curr->connections) {
        if (visited[entry.second->idx] == false) {
          visited[entry.second->idx] = true;
          ip_queue.push(entry.second);
          ip_step.push(step + 1);
          if (entry.second->name.find(end) != std::string::npos) {
            if (entry.second->name.find(":") != std::string::npos) {
              ip_mapping[i].to.push_back(entry.second->name);
            }
          }
        }
      }
    }
  }

  return ip_mapping;
}

std::vector<std::string> StreamScan::findIPs(std::string desc, bool arg) {
  std::vector<std::string> ips_found = {};

  if (!arg) {
    for (auto &ip : ips)
      if (ip.second.name.find(desc) != std::string::npos)
        if (ip.second.name.find(":") != std::string::npos)
          ips_found.push_back(ip.second.name);

  } else {
    for (auto &arg : args)
      if (arg.second.name.find(desc) != std::string::npos)
        ips_found.push_back(ips[arg.second.ip_idx].name);
  }

  return ips_found;
}

std::vector<uint32_t> StreamScan::getIPbanks(std::string ip_name,
                                             std::vector<std::string> args_desc,
                                             bool inc_stream) {
  std::vector<uint32_t> banks = {};

  using IPMap = std::unordered_map<uint32_t, ip_type>;

  auto is_ip = [&](IPMap::const_reference ip_entry) {
    if (ip_entry.second.name.find(ip_name) != std::string::npos)
      if (ip_entry.second.name.find(":") != std::string::npos)
        return true;
    return false;
  };

  auto result = std::find_if(ips.begin(), ips.end(), is_ip);

  // if found an matching ip
  if (result != ips.end()) {
    std::string ip_name = result->second.name;
    uint32_t ip_idx = result->second.idx;

    // Evaluate all args
    for (auto &arg : args) {
      if (args_desc.size() != 0) {
        bool eval = false;
        for (auto &arg_desc : args_desc) {
          if (arg.second.name.find(arg_desc) != std::string::npos) {
            eval = true;
          }
        }
        if (!eval) // if not present in desired arguments, skip
          continue;
      }
      if (arg.second.ip_idx == ip_idx) {
        if (!inc_stream && mems.find(arg.second.mem_idx) != mems.end())
          continue;

        if (std::find(banks.begin(), banks.end(), arg.second.mem_idx) ==
            banks.end())
          banks.push_back(arg.second.mem_idx);
      }
    }
  }

  return banks;
}

void StreamScan::dump_mem_section() {
  if (sections == nullptr) {
    log_debug("Memory Topology section not read");
    return;
  }

  const mem_topology *sec = sections->mem_sec;

  std::cout << "Mem Topology Dump:" << std::endl;
  for (int i = 0; i < sec->m_count; i++) {
    std::cout << "Mem " << i << " -> "
              << "Tag: " << sec->m_mem_data[i].m_tag << " | "
              << "Type: " << mem2str(sec->m_mem_data[i].m_type) << std::endl;
  }
}

void StreamScan::dump_ips_section() {
  if (sections == nullptr) {
    log_debug("IP Layout section not read");
    return;
  }

  const ip_layout *sec = sections->ips_sec;

  std::cout << "IP Layout Dump:" << std::endl;
  for (int i = 0; i < sec->m_count; i++) {
    std::cout << "IP " << i << " -> "
              << "Name: " << sec->m_ip_data[i].m_name << " | "
              << "Type: " << sec->m_ip_data[i].m_type << std::endl;
  }
}

void StreamScan::dump_con_section() {
  if (sections == nullptr) {
    log_debug("Connectivity section not read");
    return;
  }

  const connectivity *sec = sections->con_sec;
  std::cout << "Connectivity Dump:" << std::endl;
  for (int i = 0; i < sec->m_count; i++) {
    std::cout << "Con " << i << " -> "
              << "Arg Idx: " << sec->m_connection[i].arg_index << " | "
              << "IP Idx: " << sec->m_connection[i].m_ip_layout_index << " | "
              << "Data Idx: " << sec->m_connection[i].mem_data_index
              << std::endl;
  }
}
} // namespace stream_scan