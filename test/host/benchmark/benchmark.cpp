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
#include <accl_network_utils.hpp>
#include <experimental/xrt_ip.h>
#include <filesystem>
#include <fstream>
#include <json/json.h>
#include <numeric>
#include <mpi.h>
#include <regex>
#include <tclap/CmdLine.h>
#include <vector>
#include <probe.hpp>

using namespace ACCL;
using namespace accl_network_utils;
namespace fs = std::filesystem;

int rank, size;

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
  bool rsfec;
  std::vector<unsigned> counts;
  std::string xclbin;
  std::string config_file;
  fs::path result_file;
  std::vector<std::string> ips;
};

struct results_t {
  std::vector<long double> latency;
  std::vector<long double> rtt;
  std::map<unsigned, std::vector<long double>> throughput;
  std::map<unsigned, std::vector<long double>> allreduce;
};

long double benchmark_latency(ACCL::ACCL &accl, unsigned int run, options_t &options) {
  std::cerr << "Start latency benchmark (" << run << ")..." << std::endl;
  Timer timer{};

  std::cerr << "Starting timer..." << std::endl;
  timer.start();
  accl.nop();
  timer.end();
  std::cerr << "Timer end." << std::endl;

  long double time = timer.elapsed() / 1e6L;
  if (rank == 1) {
    std::cerr << "Kernel latency time: " << time * 1e3L << " ms" << std::endl;
  }
  return time;
}

long double benchmark_rtt(ACCL::ACCL &accl, unsigned int run, options_t &options) {
  const unsigned int smallest_count = 1;
  std::cerr << "Start round trip time benchmark (" << run << ")..." << std::endl;
  auto send_buf = accl.create_buffer<uint32_t>(smallest_count, dataType::int32);
  auto recv_buf = accl.create_buffer<uint32_t>(smallest_count, dataType::int32);
  Timer timer{};
  if (rank == 1) {
    send_buf->sync_to_device();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    accl.recv(*recv_buf, smallest_count, 1, 0, GLOBAL_COMM, true);
    accl.send(*recv_buf, smallest_count, 1, 1, GLOBAL_COMM, true);
  } else {
    std::cerr << "Starting timer..." << std::endl;
    timer.start();
    accl.send(*send_buf, smallest_count, 0, 0, GLOBAL_COMM, true);
    accl.recv(*recv_buf, smallest_count, 0, 1, GLOBAL_COMM, true);
    timer.end();
    std::cerr << "Timer end." << std::endl;
  }

  recv_buf->sync_from_device();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 1) {
    long double time = timer.elapsed() / 1e6L;
    std::cerr << "Round trip time: " << time * 1e3L << " ms" << std::endl;
    return time;
  } else {
    return 0.0L;
  }
}

long double benchmark_throughput(ACCL::ACCL &accl, unsigned int count, unsigned int run, options_t &options) {
  std::cerr << "Start throughput benchmark with " << count << " items (" << run << ")..." << std::endl;
  auto buf = accl.create_buffer<uint32_t>(count, dataType::int32);
  if (rank == 0) {
    buf->sync_to_device();
  }
  std::iota(buf->buffer(), buf->buffer() + count, 0U);
  Timer timer{};
  MPI_Barrier(MPI_COMM_WORLD);

  std::cerr << "Starting timer..." << std::endl;
  timer.start();
  if (rank == 0) {
    accl.send(*buf, count, 1, 0, GLOBAL_COMM, true);
  } else {
    accl.recv(*buf, count, 0, 0, GLOBAL_COMM, true);
  }
  timer.end();
  std::cerr << "Timer end." << std::endl;

  if (rank == 1) {
    buf->sync_from_device();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  long double data_size = count * 4;
  long double time = timer.elapsed() / 1e6L;
  long double bandwidth = (data_size * 8 / time) / 1e9L;
  if (rank == 1) {
    std::cerr << "data size: " << data_size / 1e6L << " MB; time: "
              << time * 1e3 << " ms; bandwidth: " << bandwidth << " Gbps"
              << std::endl;
  }
  return time;
}

long double benchmark_allreduce(ACCL::ACCL &accl, unsigned int count, unsigned int run, options_t &options) {
  std::cerr << "Start allreduce benchmark with " << count << " items (" << run << ")..." << std::endl;
  auto sendbuf = accl.create_buffer<uint32_t>(count, dataType::int32);
  auto recvbuf = accl.create_buffer<uint32_t>(count, dataType::int32);
  std::iota(sendbuf->buffer(), sendbuf->buffer() + count, 0U);
  Timer timer{};
  sendbuf->sync_to_device();
  MPI_Barrier(MPI_COMM_WORLD);

  std::cerr << "Starting timer..." << std::endl;
  timer.start();
  accl.allreduce(*sendbuf, *recvbuf, count, reduceFunction::SUM, GLOBAL_COMM, true, true);
  timer.end();
  std::cerr << "Timer end." << std::endl;

  recvbuf->sync_from_device();
  MPI_Barrier(MPI_COMM_WORLD);
  long double data_size = count * 4;
  long double time = timer.elapsed() / 1e6L;
  long double bandwidth = (data_size * 8 / time) / 1e9L;
  if (rank == 1) {
    std::cerr << "data size: " << data_size / 1e6L << " MB; time: "
              << time * 1e3 << " ms; bandwidth: " << bandwidth << " Gbps"
              << std::endl;
  }
  return time;
}

results_t start_benchmark(options_t options) {
  std::vector<rank_t> ranks;

  if (options.config_file == "") {
    ranks = generate_ranks(options.axis3, rank, size,
                           options.start_port, options.rxbuf_size);
  } else {
    ranks = generate_ranks(options.config_file, rank, options.start_port,
                           options.rxbuf_size, options.roce);
  }

  acclDesign design;
  if (options.axis3) {
    design = acclDesign::AXIS3x;
  } else if (options.udp) {
    design = acclDesign::UDP;
  } else if (options.tcp) {
    design = acclDesign::TCP;
  } else if (options.roce) {
    design = acclDesign::ROCE;
  }

  xrt::device device = xrt::device(options.device_index);
  
  auto xclbin_uuid = device.load_xclbin(options.xclbin);
  auto probe = xrt::kernel(device, xclbin_uuid, "call_probe:{probe_0}");
  ACCLProbe probe = ACCLProbe(device, probe);//create a probe object
  probe.skip(0); //skip an infinite number of calls

  std::unique_ptr<ACCL::ACCL> accl = initialize_accl(
      ranks, rank, false, design, device, options.xclbin, 16,
      options.rxbuf_size, options.segment_size, options.rsfec);

  accl->set_timeout(1e6);

  probe.disarm(); //stop skipping

  results_t results{};


  for (unsigned int run = 0; run < options.nruns; run++) {
    long double latency = benchmark_latency(*accl, run, options);
    results.latency.emplace_back(latency);
    long double rtt = benchmark_rtt(*accl, run, options);
    results.rtt.emplace_back(rtt);
  }

  for (unsigned int count : options.counts) {
    std::vector<long double> throughputs(options.nruns);
    std::vector<long double> allreduces(options.nruns);
    for (unsigned int run = 0; run < options.nruns; run++) {
      long double throughput = benchmark_throughput(*accl, count, run, options);
      throughputs[run] = throughput;
      long double allreduce = benchmark_allreduce(*accl, count, run, options);
      allreduces[run] = allreduce;
    }
    results.throughput.emplace(count, throughputs);
    results.allreduce.emplace(count, allreduces);
  }

  return results;
}

void write_results_to_file(const results_t &results, fs::path file) {
  Json::Value result;

  Json::Value latency;
  for (long double l : results.latency) {
    latency.append(static_cast<double>(l));
  }
  result["latency"] = latency;

  Json::Value rtt;
  for (auto &&r : results.rtt) {
    rtt.append(static_cast<double>(r));
  }
  result["rtt"] = rtt;

  Json::Value throughputs;
  for (auto &&item : results.throughput) {
    Json::Value throughput;
    for (auto &&t : item.second) {
      throughput.append(static_cast<double>(t));
    }
    throughputs[std::to_string(item.first)] = throughput;
  }

  result["throughput"] = throughputs;

  Json::Value allreduces;
  for (auto &&item : results.allreduce) {
    Json::Value allreduce;
    for (auto &&t : item.second) {
      allreduce.append(static_cast<double>(t));
    }
    allreduces[std::to_string(item.first)] = allreduce;
  }

  result["allreduce"] = allreduces;

  Json::StreamWriterBuilder builder;
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  std::ofstream f;
  f.open(file);
  writer->write(result, &f);
  f.close();
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
  TCLAP::ValueArg<std::string> file_arg(
      "", "results-file",
      "JSON file to write results to. {rank} will be replaced by the rank. "
      "Defaults to 'results-{rank}.json'",
      false, "results-{rank}.json", "string");
  cmd.add(file_arg);
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
  TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd,
                             false);
  TCLAP::UnlabeledMultiArg<unsigned int> count_arg(
     "count", "How many items per test", true, "counts to benchmark");
  cmd.add(count_arg);

  try {
    cmd.parse(argc, argv);
    if (hardware_arg.getValue()) {
      if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() +
          roce_arg.getValue() != 1) {
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
  opts.rsfec = rsfec_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  std::string result_file = file_arg.getValue();
  opts.result_file = std::regex_replace(result_file, std::regex("\\{rank\\}"),
                                        std::to_string(rank));
  opts.config_file = config_arg.getValue();
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
  results_t results;

  try {
    results = start_benchmark(options);
  } catch (network_error &e) {
    std::cout << "ACCL network error: " << e.what() << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    MPI_Finalize();
    return 1;
  }

  write_results_to_file(results, options.result_file);
  MPI_Finalize();
  return 0;
}
