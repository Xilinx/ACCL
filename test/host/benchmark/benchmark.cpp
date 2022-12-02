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
#
*******************************************************************************/

#include <accl.hpp>
#include <accl/timing.hpp>
#include <experimental/xrt_ip.h>
#include <fstream>
#include <json/json.h>
#include <numeric>
#include <mpi.h>
#include <roce/cmac.hpp>
#include <roce/hivenet.hpp>
#include <tclap/CmdLine.h>
#include <vector>
#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

using namespace ACCL;

// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

int rank, size;
unsigned failed_tests;
unsigned skipped_tests;

struct options_t {
  int start_port;
  unsigned int rxbuf_size;
  unsigned int segment_size;
  unsigned int nruns;
  unsigned int device_index;
  bool axis3;
  bool udp;
  bool tcp;
  bool roce;
  std::vector<unsigned> counts;
  std::string xclbin;
  std::string config_file;
  std::vector<std::string> ips;
};

bool check_arp(vnx::Networklayer &network_layer, std::vector<rank_t> &ranks,
               options_t &options) {
  std::map<unsigned, bool> ranks_checked;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    ranks_checked[i] = false;
  }

  bool sanity_check = true;
  const std::map<int, std::pair<std::string, std::string>> arp =
      network_layer.read_arp_table(size);

  std::ostringstream ss_arp;
  ss_arp << "ARP table:";

  for (const std::pair<const int, std::pair<std::string, std::string>> &elem :
       arp) {
    const unsigned index = elem.first;
    const std::pair<std::string, std::string> &entry = elem.second;
    const std::string &mac = entry.first;
    const std::string &ip = entry.second;
    ss_arp << "\n(" << index << ") " << mac << ": " << ip;

    for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
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

  if (!sanity_check) {
    return false;
  }

  unsigned hosts = 0;
  for (unsigned i = 0; i < static_cast<unsigned>(size); ++i) {
    if (ranks_checked[i]) {
      hosts += 1;
    }
  }
  if (hosts < static_cast<unsigned>(size) - 1) {
    std::cout << "Found only " << hosts << " hosts out of " << size - 1 << "!"
              << std::endl;
    return false;
  }

  return true;
}

void configure_vnx(vnx::CMAC &cmac, vnx::Networklayer &network_layer,
                   std::vector<rank_t> &ranks, options_t &options) {
  if (ranks.size() > vnx::max_sockets_size) {
    throw std::runtime_error("Too many ranks. VNX supports up to " +
                             std::to_string(vnx::max_sockets_size) +
                             " sockets.");
  }

  std::cout << "Testing UDP link status: ";

  const auto link_status = cmac.link_status();

  if (link_status.at("rx_status")) {
    std::cout << "Link successful!" << std::endl;
  } else {
    std::cout << "No link found." << std::endl;
  }

  std::ostringstream ss;

  ss << "Link interface 1 : {";
  for (const auto &elem : link_status) {
    ss << elem.first << ": " << elem.second << ", ";
  }
  ss << "}" << std::endl;

  if (!link_status.at("rx_status")) {
    // Give time for other ranks to setup link.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    exit(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "Populating socket table..." << std::endl;

  network_layer.update_ip_address(ranks[rank].ip);
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (i == static_cast<size_t>(rank)) {
      continue;
    }

    network_layer.configure_socket(i, ranks[i].ip, ranks[i].port,
                                   ranks[rank].port, true);
  }

  network_layer.populate_socket_table();

  std::cout << "Starting ARP discovery..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(4));
  MPI_Barrier(MPI_COMM_WORLD);
  network_layer.arp_discovery();
  std::cout << "Finishing ARP discovery..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  MPI_Barrier(MPI_COMM_WORLD);
  network_layer.arp_discovery();
  std::cout << "ARP discovery finished!" << std::endl;

  if (!check_arp(network_layer, ranks, options)) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    exit(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void configure_tcp(BaseBuffer &tx_buf_network, BaseBuffer &rx_buf_network,
                   xrt::kernel &network_krnl, std::vector<rank_t> &ranks,
                   options_t &options) {
  std::cout << "Configure TCP Network Kernel" << std::endl;
  tx_buf_network.sync_to_device();
  rx_buf_network.sync_to_device();

  uint local_fpga_ip = ip_encode(ranks[rank].ip);
  std::cout << "rank: " << rank << " FPGA IP: " << std::hex << local_fpga_ip
            << std::endl;

  network_krnl(local_fpga_ip, static_cast<uint32_t>(rank), local_fpga_ip,
               *(tx_buf_network.bo()), *(rx_buf_network.bo()));

  uint32_t ip_reg = network_krnl.read_register(0x010);
  uint32_t board_reg = network_krnl.read_register(0x018);
  std::cout << std::hex << "ip_reg: " << ip_reg
            << " board_reg IP: " << board_reg << std::endl;
}

void configure_roce(roce::CMAC &cmac, roce::Hivenet &hivenet,
                    std::vector<rank_t> &ranks, options_t &options) {
  uint32_t subnet_e = ip_encode(ranks[rank].ip) & 0xFFFFFF00;
  std::string subnet = ip_decode(subnet_e);
  uint32_t local_id = hivenet.get_local_id();
  std::string internal_ip = ip_decode(subnet_e + local_id);

  if (ranks[rank].ip != internal_ip) {
    throw std::runtime_error(
      "IP address set (" + ranks[rank].ip + ") mismatches with internal " +
      "hivenet IP (" + internal_ip + "). The internal ip is determined by " +
      "adding the rank (" + std::to_string(rank) + ") to the subnet (" +
      subnet + ").");
  }

  hivenet.set_ip_subnet(subnet);
  hivenet.set_mac_subnet(0x347844332211);
  cmac.set_rs_fec(true);

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Testing RoCE link status: ";

  const auto link_status = cmac.link_status();

  if (link_status.at("rx_status")) {
    std::cout << "Link successful!" << std::endl;
  } else {
    std::cout << "No link found." << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!link_status.at("rx_status")) {
    throw std::runtime_error("No link on ethernet.");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<std::string> get_ips(std::string config_file, bool local) {
  std::vector<std::string> ips;
  if (config_file == "") {
    for (int i = 0; i < size; ++i) {
      if (local) {
        ips.emplace_back("127.0.0.1");
      } else {
        ips.emplace_back("10.10.10." + std::to_string(i));
      }
    }
  } else {
    Json::Value config;
    std::ifstream config_file_stream(config_file);
    config_file_stream >> config;
    Json::Value ip_config = config["ips"];
    for (int i = 0; i < size; ++i) {
      std::string ip = ip_config[i].asString();
      if (ip == "") {
        throw std::runtime_error("No ip for rank " + std::to_string(i) +
                                 " in config file.");
      }
      ips.push_back(ip);
    }
  }

  return ips;
}

void benchmark(ACCL::ACCL &accl, unsigned int count, unsigned int run, options_t &options) {
  std::cerr << "Start benchmark with " << count << " items (" << run << ")..." << std::endl;
  auto buf = accl.create_buffer<uint32_t>(count, dataType::int32);
  std::iota(buf->buffer(), buf->buffer() + count, 0U);
  Timer timer{};
  MPI_Barrier(MPI_COMM_WORLD);

  std::cerr << "Starting timer..." << std::endl;
  timer.start();
  if (rank == 0) {
    accl.send(*buf, count, 1, 0);
  } else {
    accl.recv(*buf, count, 0, 0);
  }
  timer.end();
  std::cerr << "Timer end." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  long double data_size = (count * 4) / 1e6L;
  long double time = timer.elapsed() / 1e3L;
  long double bandwidth = ((count * 32) / (timer.elapsed() / 1e6L)) / 1e9L;
  std::cout << "data size: " << data_size << " MB; time: " << time
            << " ms; bandwidth: " << bandwidth << " Gbps" << std::endl;
}

int start_test(options_t options) {
  std::vector<rank_t> ranks = {};
  failed_tests = 0;
  skipped_tests = 0;
  options.ips =
      get_ips(options.config_file, options.axis3);
  for (int i = 0; i < size; ++i) {
    rank_t new_rank = {options.ips[i], options.start_port + i, i,
                       options.rxbuf_size};
    ranks.emplace_back(new_rank);
  }

  std::unique_ptr<ACCL::ACCL> accl;
  std::unique_ptr<ACCL::BaseBuffer> tx_buf_network;
  std::unique_ptr<ACCL::BaseBuffer> rx_buf_network;

  xrt::device device = xrt::device(options.device_index);;

  networkProtocol protocol;

  std::string cclo_id;
  if (options.axis3) {
    cclo_id = std::to_string(rank);
  } else {
    cclo_id = "0";
  }
  auto xclbin_uuid = device.load_xclbin(options.xclbin);
  auto cclo_ip = xrt::ip(device, xclbin_uuid,
                          "ccl_offload:{ccl_offload_" + cclo_id + "}");
  auto hostctrl_ip = xrt::kernel(device, xclbin_uuid,
                                  "hostctrl:{hostctrl_" + cclo_id + "_0}",
                                  xrt::kernel::cu_access_mode::exclusive);

  int devicemem;
  std::vector<int> rxbufmem;
  int networkmem;
  if (options.axis3) {
    devicemem = rank * 6;
    rxbufmem = {rank * 6 + 1};
    networkmem = rank * 6 + 2;
  } else if (options.roce) {
    devicemem = 3;
    rxbufmem = {4};
    networkmem = 6;
  } else {
    devicemem = 0;
    rxbufmem = {1};
    networkmem = 6;
  }

  protocol = options.tcp ? networkProtocol::TCP : networkProtocol::UDP;

  if (options.udp) {
    auto cmac = vnx::CMAC(xrt::ip(device, xclbin_uuid, "cmac_0:{cmac_0}"));
    auto network_layer = vnx::Networklayer(
        xrt::ip(device, xclbin_uuid, "networklayer:{networklayer_0}"));

    configure_vnx(cmac, network_layer, ranks, options);
  } else if (options.tcp) {
    tx_buf_network = std::unique_ptr<BaseBuffer>(new FPGABuffer<int8_t>(
        64 * 1024 * 1024, dataType::int8, device, networkmem));
    rx_buf_network = std::unique_ptr<BaseBuffer>(new FPGABuffer<int8_t>(
        64 * 1024 * 1024, dataType::int8, device, networkmem));
    auto network_krnl =
        xrt::kernel(device, xclbin_uuid, "network_krnl:{network_krnl_0}",
                    xrt::kernel::cu_access_mode::exclusive);
    configure_tcp(*tx_buf_network, *rx_buf_network, network_krnl, ranks,
                  options);
  } else if (options.roce) {
    auto cmac = roce::CMAC(xrt::ip(device, xclbin_uuid, "cmac_0:{cmac_0}"));
    auto hivenet = roce::Hivenet(
        xrt::ip(device, xclbin_uuid, "HiveNet_kernel_0:{networklayer_0}"),
        rank);

    configure_roce(cmac, hivenet, ranks, options);
  }

  accl = std::make_unique<ACCL::ACCL>(ranks, rank, device, cclo_ip,
                                      hostctrl_ip, devicemem, rxbufmem,
                                      protocol, 16, options.rxbuf_size,
                                      options.segment_size);

  if (protocol == networkProtocol::TCP) {
    MPI_Barrier(MPI_COMM_WORLD);
    accl->open_port();
    MPI_Barrier(MPI_COMM_WORLD);
    accl->open_con();
  }

  accl->set_timeout(1e6);

  MPI_Barrier(MPI_COMM_WORLD);
  accl->nop();
  MPI_Barrier(MPI_COMM_WORLD);
  for (unsigned int count : options.counts) {
    for (unsigned int run = 0; run < options.nruns; run++) {
      benchmark(*accl, count, run, options);
    }
  }

  return 0;
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
  TCLAP::ValueArg<unsigned int> bufsize_arg("b", "rxbuf-size",
                                            "How many KB per RX buffer", false,
                                            1, "positive integer");
  cmd.add(bufsize_arg);
  TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
  TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                false);
  TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                             false);
  TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
  TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
  TCLAP::SwitchArg roce_arg("r", "roce", "Use RoCE hardware setup", cmd, false);
  TCLAP::ValueArg<std::string> xclbin_arg(
      "x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
      "accl.xclbin", "file");
  cmd.add(xclbin_arg);
  TCLAP::ValueArg<uint16_t> device_index_arg(
      "i", "device-index", "device index of FPGA if hardware mode is used",
      false, 0, "positive integer");
  cmd.add(device_index_arg);
  TCLAP::ValueArg<std::string> config_arg("c", "config",
                                          "Config file containing IP mapping",
                                          false, "", "JSON file");
  cmd.add(config_arg);
  TCLAP::UnlabeledMultiArg<unsigned int> count_arg(
     "count", "How many items per test", true, "counts to benchmark");
  cmd.add(count_arg);

  try {
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
      if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() +
              roce_arg.getValue() !=
          1) {
        throw std::runtime_error("When using hardware, specify one of axis3, "
                                 "tcp, udp, or roce mode, but not both.");
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
  opts.counts = count_arg.getValue();
  opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
  opts.segment_size = opts.rxbuf_size;
  opts.nruns = nruns_arg.getValue();
  opts.axis3 = axis3_arg.getValue();
  opts.udp = udp_arg.getValue();
  opts.tcp = tcp_arg.getValue();
  opts.roce = roce_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.config_file = config_arg.getValue();
  return opts;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  options_t options = parse_options(argc, argv);

  std::ostringstream stream;
  stream << "rank " << rank << " size " << size
         << std::endl;
  std::cout << stream.str();

  int errors = start_test(options);

  MPI_Finalize();
  return errors;
}
