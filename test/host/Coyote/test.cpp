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

#include "accl.hpp"
#include <cstdlib>
#include <functional>
#include <mpi.h>
#include <random>
#include <sstream>
#include <tclap/CmdLine.h>
#include <vector>
#include <fstream>
#include <arpa/inet.h>
#include <string>

using namespace ACCL;

// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

#define FREQ 250
#define MAX_PKT_SIZE 4096

int mpi_rank, mpi_size;
unsigned failed_tests;
unsigned skipped_tests;

// leave options be for now to avoid any argument parsing issues

struct options_t
{
	int start_port;
	unsigned int rxbuf_size;
	unsigned int seg_size;
	unsigned int count;
	unsigned int nruns;
	unsigned int device_index;
	unsigned int num_rxbufmem;
	unsigned int test_mode;
	bool debug;
	bool hardware;
	bool axis3;
	bool udp;
	bool tcp;
	unsigned int host;
	std::string xclbin;
	std::string fpgaIP;
};

struct timestamp_t
{
	uint64_t cmdSeq;
	uint64_t scenario;
	uint64_t len;
	uint64_t comm;
	uint64_t root_src_dst;
	uint64_t function;
	uint64_t msg_tag;
	uint64_t datapath_cfg;
	uint64_t compression_flags;
	uint64_t stream_flags;
	uint64_t addra_l;
	uint64_t addra_h;
	uint64_t addrb_l;
	uint64_t addrb_h;
	uint64_t addrc_l;
	uint64_t addrc_h;
	uint64_t cmdTimestamp;
	uint64_t cmdEnd;
	uint64_t stsSeq;
	uint64_t sts;
	uint64_t stsTimestamp;
	uint64_t stsEnd;
};

//******************************
//**  XCC Operations          **
//******************************
// Housekeeping
#define ACCL_CONFIG 0
// Primitives
#define ACCL_COPY 1
#define ACCL_COMBINE 2
#define ACCL_SEND 3
#define ACCL_RECV 4
// Collectives
#define ACCL_BCAST 5
#define ACCL_SCATTER 6
#define ACCL_GATHER 7
#define ACCL_REDUCE 8
#define ACCL_ALLGATHER 9
#define ACCL_ALLREDUCE 10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER 12
#define ACCL_ALLTOALL 13

// ACCL_CONFIG SUBFUNCTIONS
#define HOUSEKEEP_SWRST 0
#define HOUSEKEEP_PKTEN 1
#define HOUSEKEEP_TIMEOUT 2
#define HOUSEKEEP_OPEN_PORT 3
#define HOUSEKEEP_OPEN_CON 4
#define HOUSEKEEP_SET_STACK_TYPE 5
#define HOUSEKEEP_SET_MAX_SEGMENT_SIZE 6
#define HOUSEKEEP_CLOSE_CON 7

std::string format_log(std::string collective, options_t options, double time, double tput)
{
	std::string log_str = collective + "," + std::to_string(mpi_size) + "," + std::to_string(mpi_rank) + "," + std::to_string(options.num_rxbufmem) + "," + std::to_string(options.count * sizeof(float)) + "," + std::to_string(options.rxbuf_size) + "," + std::to_string(options.rxbuf_size) + "," + std::to_string(MAX_PKT_SIZE) + "," + std::to_string(time) + "," + std::to_string(tput) + "," + std::to_string(options.host);
	return log_str;
}

inline void swap_endianness(uint32_t *ip)
{
	uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
	*ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
		  (ip_bytes[0] << 24);
}

uint32_t _ip_encode(std::string ip)
{
	struct sockaddr_in sa;
	inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
	swap_endianness(&sa.sin_addr.s_addr);
	return sa.sin_addr.s_addr;
}

std::string ip_decode(uint32_t ip)
{
	char buffer[INET_ADDRSTRLEN];
	struct in_addr sa;
	sa.s_addr = ip;
	swap_endianness(&sa.s_addr);
	inet_ntop(AF_INET, &sa, buffer, INET_ADDRSTRLEN);
	return std::string(buffer, INET_ADDRSTRLEN);
}

void test_debug(std::string message, options_t &options)
{
	if (options.debug)
	{
		std::cerr << message << std::endl;
	}
}

void check_usage(int argc, char *argv[]) {}

std::string prepend_process()
{
	return "[process " + std::to_string(mpi_rank) + "] ";
}

template <typename T>
bool is_close(T a, T b, double rtol = 1e-5, double atol = 1e-8)
{
	// std::cout << abs(a - b) << " <= " << (atol + rtol * abs(b)) << "? " <<
	// (abs(a - b) <= (atol + rtol * abs(b))) << std::endl;
	return abs(a - b) <= (atol + rtol * abs(b));
}

template <typename T>
static void random_array(T *data, size_t count)
{
	std::uniform_real_distribution<T> distribution(-1000, 1000);
	std::mt19937 engine;
	auto generator = std::bind(distribution, engine);
	for (size_t i = 0; i < count; ++i)
	{
		data[i] = generator();
	}
}

template <typename T>
std::unique_ptr<T> random_array(size_t count)
{
	std::unique_ptr<T> data(new T[count]);
	random_array(data.get(), count);
	return data;
}


options_t parse_options(int argc, char *argv[])
{
	try
	{
		TCLAP::CmdLine cmd("Test ACCL C++ driver");
		TCLAP::ValueArg<unsigned int> nruns_arg("n", "nruns",
												"How many times to run each test",
												false, 1, "positive integer");
		cmd.add(nruns_arg);
		TCLAP::ValueArg<uint16_t> start_port_arg(
			"s", "start-port", "Start of range of ports usable for sim", false, 5005,
			"positive integer");
		cmd.add(start_port_arg);
		TCLAP::ValueArg<uint32_t> count_arg("c", "count", "How many element per buffer",
											false, 16, "positive integer");
		cmd.add(count_arg);
		TCLAP::ValueArg<uint16_t> bufsize_arg("b", "rxbuf-size",
											  "How many KB per RX buffer", false, 4096,
											  "positive integer");
		cmd.add(bufsize_arg);
		TCLAP::ValueArg<uint32_t> seg_arg("g", "max_segment_size",
										  "Maximum segmentation size in KB (should be samller than Max DMA transaction)", false, 4096,
										  "positive integer");
		cmd.add(seg_arg);
		TCLAP::ValueArg<uint16_t> num_rxbufmem_arg("m", "num_rxbufmem",
												   "Number of memory banks used for rxbuf", false, 2,
												   "positive integer");
		cmd.add(num_rxbufmem_arg);
		TCLAP::ValueArg<uint16_t> test_mode_arg("y", "test_mode",
												"Test mode, by default run all the collective tests", false, 0,
												"integer");
		cmd.add(test_mode_arg);
		TCLAP::ValueArg<uint16_t> host_arg("z", "host_buffer",
												"Enable host buffer mode with 1", false, 0,
												"integer");
		cmd.add(host_arg);
		TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
		TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd, false);
		TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd, false);
		TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
		TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
		TCLAP::SwitchArg userkernel_arg("k", "userkernel", "Enable user kernel(by default vadd kernel)", cmd, false);
		TCLAP::ValueArg<std::string> xclbin_arg(
			"x", "xclbin", "xclbin of accl driver if hardware mode is used", false,
			"accl.xclbin", "file");
		cmd.add(xclbin_arg);
		TCLAP::ValueArg<std::string> fpgaIP_arg(
			"l", "ipList", "ip list of FPGAs if hardware mode is used", false,
			"fpga", "file");
		cmd.add(fpgaIP_arg);
		TCLAP::ValueArg<uint16_t> device_index_arg(
			"i", "device-index", "device index of FPGA if hardware mode is used",
			false, 0, "positive integer");
		cmd.add(device_index_arg);
		cmd.parse(argc, argv);
		if (hardware_arg.getValue())
		{
			if (axis3_arg.getValue())
			{
				if (udp_arg.getValue() || tcp_arg.getValue())
				{
					throw std::runtime_error("When using hardware axis3 mode, tcp or udp can not be used.");
				}
				std::cout << "Hardware axis3 mode" << std::endl;
			}
			if (udp_arg.getValue())
			{
				if (axis3_arg.getValue() || tcp_arg.getValue())
				{
					throw std::runtime_error("When using hardware udp mode, tcp or axis3 can not be used.");
				}
				std::cout << "Hardware udp mode" << std::endl;
			}
			if (tcp_arg.getValue())
			{
				if (axis3_arg.getValue() || udp_arg.getValue())
				{
					throw std::runtime_error("When using hardware tcp mode, udp or axis3 can not be used.");
				}
				std::cout << "Hardware tcp mode" << std::endl;
			}
			if ((axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue()) == false)
			{
				throw std::runtime_error("When using hardware, specify either axis3 or tcp or"
										 "udp mode.");
			}
		}

		options_t opts;
		opts.start_port = start_port_arg.getValue();
		opts.count = count_arg.getValue();
		opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
		opts.seg_size = seg_arg.getValue() * 1024;		 // convert to bytes
		opts.num_rxbufmem = num_rxbufmem_arg.getValue();
		opts.nruns = nruns_arg.getValue();
		opts.debug = debug_arg.getValue();
		opts.host = host_arg.getValue();
		opts.hardware = hardware_arg.getValue();
		opts.axis3 = axis3_arg.getValue();
		opts.udp = udp_arg.getValue();
		opts.tcp = tcp_arg.getValue();
		opts.test_mode = test_mode_arg.getValue();
		opts.device_index = device_index_arg.getValue();
		opts.xclbin = xclbin_arg.getValue();
		opts.fpgaIP = fpgaIP_arg.getValue();

		std::cout << "count:" << opts.count << " rxbuf_size:" << opts.rxbuf_size << " seg_size:" << opts.seg_size << " num_rxbufmem:" << opts.num_rxbufmem << std::endl;
		return opts;
	}
	catch (std::exception &e)
	{
		if (mpi_rank == 0)
		{
			std::cout << "Error: " << e.what() << std::endl;
		}

		MPI_Finalize();
		exit(1);
	}
}

void test_sendrcv(ACCL::ACCL &accl, options_t &options) {
  	std::cout << "Start send recv test..." << std::endl<<std::flush;
	// do the send recv test here
	int bufsize = options.count;
	auto op_buf = accl.create_coyotebuffer<int>(bufsize, dataType::int32);
	
	// rank 0 initializes the buffer with numbers, rank1 with -1
	for (int i = 0; i < bufsize; i++) op_buf.get()->buffer()[i] = (mpi_rank == 0) ? i : -1;
	
	if (options.host == 0){ op_buf->sync_to_device(); }

	MPI_Barrier(MPI_COMM_WORLD);
	
	double durationUs = 0.0;
  	double tput = 0.0;
  	auto start = std::chrono::high_resolution_clock::now();

	if (mpi_rank == 0) {
		// send
		accl.send(*op_buf, bufsize, 1, TAG_ANY, GLOBAL_COMM, true); // most default send from 0 to 1
	} else if (mpi_rank == 1) {
		// receive
		accl.recv(*op_buf, bufsize, 0, TAG_ANY, GLOBAL_COMM, true); // most default recv to 1 from 0
	}
	
	auto end = std::chrono::high_resolution_clock::now();
  	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
	tput = (options.count*sizeof(dataType::int32)*8.0)/(durationUs*1000.0);

	MPI_Barrier(MPI_COMM_WORLD);

	if (options.host == 0){ op_buf->sync_from_device(); }
	std::cerr << "Rank " << mpi_rank << " passed barrier after send recv test!" << std::endl;
	
	if (mpi_rank == 0 || mpi_rank == 1){
		accl_log(mpi_rank, format_log("sendrecv", options, durationUs, tput));
	}

	int errors = 0;
	if (mpi_rank == 1)
	{
		for (int i = 0; i < bufsize; i++) {
			unsigned int res = op_buf.get()->buffer()[i];
			unsigned int ref = i;
			if (res != ref) {
			std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
							std::to_string(res) + " != " + std::to_string(ref) + ")"
						<< std::endl;
			errors += 1;
			}
		}
	}
	
	if (errors > 0) {
		std::cout << std::to_string(errors) + " errors!" << std::endl;
		failed_tests++;
	} else {
		std::cout << "Test is successful!" << std::endl;
	}
}


void test_bcast(ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start bcast test with root " + std::to_string(root) + " ..."
				<< std::endl<<std::flush;
	MPI_Barrier(MPI_COMM_WORLD);
	unsigned int count = options.count;
	auto op_buf = accl.create_coyotebuffer<int>(count, dataType::int32);
	
	// rank root initializes the buffer with numbers, other ranks with -1
	for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = (mpi_rank == root) ? i : -1;

	if (options.host == 0){ op_buf->sync_to_device(); }

	if (mpi_rank == root) {
		test_debug("Broadcasting data from " + std::to_string(mpi_rank) + "...",
				options);
	} else {
		test_debug("Getting broadcast data from " + std::to_string(root) + "...",
				options);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double durationUs = 0.0;
	auto start = std::chrono::high_resolution_clock::now();
	accl.bcast(*op_buf, count, root, GLOBAL_COMM, true, true);
	auto end = std::chrono::high_resolution_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

	if (options.host == 0){ op_buf->sync_from_device(); }

	accl_log(mpi_rank, format_log("bcast", options, durationUs, 0));

	if (mpi_rank != root) {
		int errors = 0;
		for (int i = 0; i < count; i++) {
			unsigned int res = op_buf.get()->buffer()[i];
			unsigned int ref = i;
			if (res != ref) {
				// std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
				// 				std::to_string(res) + " != " + std::to_string(ref) +
				// 				")"
				// 		<< std::endl;
				errors += 1;
			}
		}

		if (errors > 0) {
			std::cout << std::to_string(errors) + " errors!" << std::endl;
			failed_tests++;
		} else {
			std::cout << "Test is successful!" << std::endl;
		}
	}

}

void test_scatter(ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start scatter test with root " + std::to_string(root) + " ..."
				<< std::endl;
	unsigned int count = options.count;
	auto op_buf = accl.create_coyotebuffer<int>(count * mpi_size, dataType::int32);
	auto res_buf = accl.create_coyotebuffer<int>(count, dataType::int32);

	// op buf initialized with i and res buf initialized with -1
	for (int i = 0; i < count * mpi_size; i++) op_buf.get()->buffer()[i] = i;
	for (int i = 0; i < count; i++) res_buf.get()->buffer()[i] = -1;

	if (options.host == 0){ op_buf->sync_to_device(); }
	if (options.host == 0){ res_buf->sync_to_device(); }

	test_debug("Scatter data from " + std::to_string(mpi_rank) + "...", options);
	
	MPI_Barrier(MPI_COMM_WORLD);
	double durationUs = 0.0;
	auto start = std::chrono::high_resolution_clock::now();
	accl.scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true);

	auto end = std::chrono::high_resolution_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

	if (options.host == 0){ res_buf->sync_from_device(); }

	accl_log(mpi_rank, format_log("scatter", options, durationUs, 0));

	int errors = 0;
	for (unsigned int i = 0; i < count; ++i) {
		unsigned int res = res_buf.get()->buffer()[i]; 
		unsigned int ref = op_buf.get()->buffer()[i + mpi_rank * count];
		if (res != ref) {
		std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
		                 std::to_string(res) + " != " + std::to_string(ref) + ")"
		          << std::endl;
		errors += 1;
		}
	}

	if (errors > 0) {
		std::cout << std::to_string(errors) + " errors!" << std::endl;
		failed_tests++;
	} else {
		std::cout << "Test is successful!" << std::endl;
	}

}


void test_gather(ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start gather test with root " + std::to_string(root) + "..."
				<< std::endl;
	unsigned int count = options.count;

	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = mpi_rank*count + i;
	if (options.host == 0){ op_buf->sync_to_device(); }

	std::unique_ptr<ACCL::Buffer<float>> res_buf;
	if (mpi_rank == root) {
		res_buf = accl.create_coyotebuffer<float>(count * mpi_size, dataType::float32);
		if (options.host == 0){ res_buf->sync_to_device(); }
	} else {
		res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
	}

	test_debug("Gather data from " + std::to_string(mpi_rank) + "...", options);

	MPI_Barrier(MPI_COMM_WORLD);
	double durationUs = 0.0;
	auto start = std::chrono::high_resolution_clock::now();

	accl.gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true);

	auto end = std::chrono::high_resolution_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

	if (mpi_rank == root){
		if (options.host == 0){ res_buf->sync_from_device(); }
	}
	
	accl_log(mpi_rank, format_log("gather", options, durationUs, 0));

	if (mpi_rank == root) {
		int errors = 0;
		for (unsigned int j = 0; j < mpi_size; ++j) {
			for (size_t i = 0; i < count; i++)
			{
				float res = res_buf.get()->buffer()[j*count+i];
				float ref = j*count+i;
				if (res != ref) {
					// std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
					// 				std::to_string(res) + " != " + std::to_string(ref) +
					// 				")"
					// 		<< std::endl;
					errors += 1;
				}
			}
		}

		if (errors > 0) {
			std::cout << std::to_string(errors) + " errors!" << std::endl;
			failed_tests++;
		} else {
			std::cout << "Test is successful!" << std::endl;
		}
	}

}


void test_allgather(ACCL::ACCL &accl, options_t &options) {
	std::cout << "Start allgather test..." << std::endl;
	unsigned int count = options.count;

	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = mpi_rank*count + i;
	if (options.host == 0){ op_buf->sync_to_device(); }

	std::unique_ptr<ACCL::Buffer<float>> res_buf;
	res_buf = accl.create_coyotebuffer<float>(count * mpi_size, dataType::float32);
	if (options.host == 0){ res_buf->sync_to_device(); }


	test_debug("Gathering data...", options);

	MPI_Barrier(MPI_COMM_WORLD);
	double durationUs = 0.0;
	auto start = std::chrono::high_resolution_clock::now();

	accl.allgather(*op_buf, *res_buf, count, GLOBAL_COMM, true, true);

	auto end = std::chrono::high_resolution_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

	if (options.host == 0){ res_buf->sync_from_device(); }

	accl_log(mpi_rank, format_log("allgather", options, durationUs, 0));

	int errors = 0;
	for (unsigned int j = 0; j < mpi_size; ++j) {
		for (size_t i = 0; i < count; i++)
		{
			float res = res_buf.get()->buffer()[j*count+i];
			float ref = j*count+i;
			if (res != ref) {
				// std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
				// 				std::to_string(res) + " != " + std::to_string(ref) +
				// 				")"
				// 		<< std::endl;
				errors += 1;
			}
		}
	}

	if (errors > 0) {
		std::cout << std::to_string(errors) + " errors!" << std::endl;
		failed_tests++;
	} else {
		std::cout << "Test is successful!" << std::endl;
	}

}

void test_reduce(ACCL::ACCL &accl, options_t &options, int root,
                 reduceFunction function) {
  std::cout << "Start reduce test with root " + std::to_string(root) +
                   " and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
  auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
  for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = i;

  if (options.host == 0){ op_buf->sync_to_device(); }

  test_debug("Reduce data to " + std::to_string(root) + "...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  accl.reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, true, true);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(mpi_rank, format_log("reduce", options, durationUs, 0));

  if (mpi_rank == root) {
    int errors = 0;

    for (unsigned int i = 0; i < count; ++i) {
      float res = res_buf.get()->buffer()[i];
      float ref = op_buf.get()->buffer()[i] * mpi_size;

      if (res != ref) {
        // std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
        //                  std::to_string(res) + " != " + std::to_string(ref) +
        //                  ")"
        //           << std::endl;
        errors += 1;
      }
    }

    if (errors > 0) {
      std::cout << std::to_string(errors) + " errors!" << std::endl;
      failed_tests++;
    } else {
      std::cout << "Test is successful!" << std::endl;
    }
  }
}

void test_allreduce(ACCL::ACCL &accl, options_t &options,
                    reduceFunction function) {
  std::cout << "Start allreduce test and reduce function " +
                   std::to_string(static_cast<int>(function)) + "..."
            << std::endl;
  unsigned int count = options.count;
  auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
  auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
  for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = i;

  test_debug("Reducing data...", options);

  MPI_Barrier(MPI_COMM_WORLD);
  double durationUs = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  accl.allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, true, true);

  auto end = std::chrono::high_resolution_clock::now();
  durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);

  accl_log(mpi_rank, format_log("allreduce", options, durationUs, 0));

  int errors = 0;

  for (unsigned int i = 0; i < count; ++i) {
    float res = res_buf.get()->buffer()[i];
    float ref = op_buf.get()->buffer()[i] * mpi_size;

    if (res != ref) {
      // std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
      //                  std::to_string(res) + " != " + std::to_string(ref) + ")"
      //           << std::endl;
      errors += 1;
    }
  }

  if (errors > 0) {
    std::cout << std::to_string(errors) + " errors!" << std::endl;
    failed_tests++;
  } else {
    std::cout << "Test is successful!" << std::endl;
  }

}

void test_accl_base(options_t options)
{
	std::cout << "Testing ACCL base functionality..." << std::endl;

	// initialize ACCL
	std::vector<ACCL::rank_t> ranks;
	int local_rank = mpi_rank;
	failed_tests = 0;
	
	// load ip addresses for targets
	std::ifstream myfile;
	myfile.open(options.fpgaIP);
	if (!myfile.is_open())
	{
		perror("Error open fpgaIP file");
		exit(EXIT_FAILURE);
	}
	std::vector<std::string> ipList;
	for (int i = 0; i < mpi_size; ++i)
	{
		std::string ip;
		if (options.hardware && !options.axis3)
		{
			ip = "10.10.10." + std::to_string(i);
			getline(myfile, ip);
			std::cout << ip << std::endl;
			ipList.push_back(ip);
		}
		else
		{
			ip = "127.0.0.1";
		}
		rank_t new_rank = {ip, options.start_port + i, i, options.rxbuf_size};
		ranks.emplace_back(new_rank);
	}
	
	// more complex initialization now, since the basics seem to work
	std::unique_ptr<ACCL::ACCL> accl;
	// construct CoyoteDevice out here already, since it is necessary for creating buffers
	// before the ACCL instance exists.
	ACCL::CoyoteDevice* device = new ACCL::CoyoteDevice();
	
	if (options.hardware)
	{
		if (options.udp)
		{
			debug("ERROR: we don't support UDP for now!!!");
			exit(1);
		}
		else if (options.tcp)
		{
			uint localFPGAIP = _ip_encode(ipList[mpi_rank]);
			std::cout << "rank: " << mpi_rank << " FPGA IP: " << std::hex << localFPGAIP << std::endl;

		}

		MPI_Barrier(MPI_COMM_WORLD);

		accl = std::make_unique<ACCL::ACCL>(device,
			ranks, mpi_rank,
			options.udp ? networkProtocol::UDP : networkProtocol::TCP,
			mpi_size, options.rxbuf_size, options.seg_size);

		if (options.tcp)
		{
			debug("Starting connections to communicator ranks");
			debug("Opening ports to communicator ranks");
			accl->open_port();
			MPI_Barrier(MPI_COMM_WORLD);
			debug("Starting session to communicator ranks");
			accl->open_con();
			debug(accl->dump_communicator());
		}

		MPI_Barrier(MPI_COMM_WORLD);

	} else {
		debug("unsupported situation!!!");
		exit(1);
	}
	
	// accl->set_timeout(1e6); // the same as in the original
	// std::cout << "set_timeout done." << std::endl;
	
	// MPI_Barrier(MPI_COMM_WORLD);
	
	// accl->nop();
	// std::cout << "nop done." << std::endl;
	
	
	std::cerr << "Rank " << mpi_rank << " passed last barrier before test!" << std::endl << std::flush;

	MPI_Barrier(MPI_COMM_WORLD);
	
	if(options.test_mode == ACCL_SEND || options.test_mode == 0){
		test_sendrcv(*accl, options);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_BCAST || options.test_mode == 0){
		test_bcast(*accl, options, 0);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_SCATTER || options.test_mode == 0){
		test_scatter(*accl, options, 0);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_GATHER || options.test_mode == 0){
		test_gather(*accl, options, 0);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_ALLGATHER || options.test_mode == 0){
		test_allgather(*accl, options);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_REDUCE || options.test_mode == 0){
		int root = 0;
		test_reduce(*accl, options, root, reduceFunction::SUM);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(options.test_mode == ACCL_ALLREDUCE || options.test_mode == 0){
		int root = 0;
		test_allreduce(*accl, options, reduceFunction::SUM);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (failed_tests == 0){
		std::cout << "\nACCL base functionality test completed successfully!\n" << std::endl;
		return;
	}
	else {
		std::cout << "\nERROR: ACCL base functionality test failed!\n" << std::endl;
		return;
	}
	
}



template <typename T>
struct aligned_allocator {
  using value_type = T;
  T* allocate(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num) {
    free(p);
  }
};


int main(int argc, char *argv[])
{
	std::cout << "Argumnents: ";
	for (int i = 0; i < argc; i++) std::cout << "'" << argv[i] << "' "; std::cout << std::endl;
	std::cout << "Running ACCL test in coyote..." << std::endl;
	std::cout << "Initializing MPI..." << std::endl;
	MPI_Init(&argc, &argv);
	
	std::cout << "Reading MPI rank and size values..." << std::endl;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	std::cout << "Parsing options" << std::endl;
	options_t options = parse_options(argc, argv);
	
	std::cout << "Getting MPI Processor name..." << std::endl;
	int len;
	char name[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(name, &len);

	std::ostringstream stream;
	stream << prepend_process() << "rank " << mpi_rank << " size " << mpi_size << " " << name
		   << std::endl;
	std::cout << stream.str();

	MPI_Barrier(MPI_COMM_WORLD);

	test_accl_base(options);

	MPI_Barrier(MPI_COMM_WORLD);
	
	std::cout << "Finalizing MPI..." << std::endl;
	MPI_Finalize();
	std::cout << "Done. Terminating..." << std::endl;
	return 0;
}