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
#include <cstring>
#ifdef COYOTE_HSA_SUPPORT
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsakmt/hsakmt.h>
#include "mapping.hpp"
#endif
#include <signal.h>

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
	bool rdma;
	unsigned int host;
	unsigned int protoc;
	std::string xclbin;
	std::string fpgaIP;
	std::string output_file;
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

#define ACCL_COPY_FPGA 14
#define ACCL_COPY_NODMA 15
#define ACCL_MAPPING 16
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
	std::string host_str;
	std::string protoc_str;
	std::string stack_str;
	if(options.host == 1){
		host_str = "host";
	} else{
		host_str = "device";
	}
	if(options.protoc == 0){
		protoc_str = "eager";
	} else if (options.protoc == 1){
		protoc_str = "rndzvs";
	}
	if(options.tcp){
		stack_str = "tcp";
	} else if (options.rdma) {
		stack_str = "rdma";
	}
	std::string log_str = collective + "," + std::to_string(mpi_size) + "," + std::to_string(mpi_rank) + "," + std::to_string(options.num_rxbufmem) + "," + std::to_string(options.count * sizeof(float)) + "," + std::to_string(options.rxbuf_size) + "," + std::to_string(options.rxbuf_size) + "," + std::to_string(MAX_PKT_SIZE) + "," + std::to_string(time) + "," + std::to_string(tput) + "," + host_str + "," + protoc_str + "," + stack_str;
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
		TCLAP::ValueArg<std::string> output_file_arg("o", "output_file",
												"Output file for .csv data", false, "output.csv",
												"file");
		cmd.add(output_file_arg);
		TCLAP::ValueArg<uint16_t> host_arg("z", "host_buffer",
												"Enable host buffer mode with 1", false, 0,
												"integer");
		cmd.add(host_arg);
		TCLAP::ValueArg<uint16_t> protoc_arg("p", "protocol",
												"Eager Protocol with 0 and Rendezvous with 1", false, 0,
												"integer");
		cmd.add(protoc_arg);
		TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
		TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd, false);
		TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd, false);
		TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
		TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
		TCLAP::SwitchArg rdma_arg("r", "rdma", "Use RDMA hardware setup", cmd, false);
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
				if (udp_arg.getValue() || tcp_arg.getValue() || rdma_arg.getValue())
				{
					throw std::runtime_error("When using hardware axis3 mode, tcp or rdma or udp can not be used.");
				}
				std::cout << "Hardware axis3 mode" << std::endl;
			}
			if (udp_arg.getValue())
			{
				if (axis3_arg.getValue() || tcp_arg.getValue() || rdma_arg.getValue())
				{
					throw std::runtime_error("When using hardware udp mode, tcp or rdma or axis3 can not be used.");
				}
				std::cout << "Hardware udp mode" << std::endl;
			}
			if (tcp_arg.getValue())
			{
				if (axis3_arg.getValue() || udp_arg.getValue() || rdma_arg.getValue())
				{
					throw std::runtime_error("When using hardware tcp mode, udp or rdma or axis3 can not be used.");
				}
				std::cout << "Hardware tcp mode" << std::endl;
			}
			if (rdma_arg.getValue())
			{
				if (axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue())
				{
					throw std::runtime_error("When using hardware rdma mode, udp or tcp or axis3 can not be used.");
				}
				std::cout << "Hardware rdma mode" << std::endl;
			}
			if ((axis3_arg.getValue() || udp_arg.getValue() || tcp_arg.getValue() || rdma_arg.getValue()) == false)
			{
				throw std::runtime_error("When using hardware, specify either axis3 or tcp or"
										 "udp or rdma mode.");
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
		opts.rdma = rdma_arg.getValue();
		opts.test_mode = test_mode_arg.getValue();
		opts.device_index = device_index_arg.getValue();
		opts.xclbin = xclbin_arg.getValue();
		opts.fpgaIP = fpgaIP_arg.getValue();
		opts.protoc = protoc_arg.getValue();
		opts.output_file = output_file_arg.getValue();

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


void exchange_qp(unsigned int master_rank, unsigned int slave_rank, unsigned int local_rank, std::vector<fpga::ibvQpConn*> &ibvQpConn_vec, std::vector<rank_t> &ranks)
{
  	
	if (local_rank == master_rank)
	{
		std::cout<<"Local rank "<<local_rank<<" sending local QP to remote rank "<<slave_rank<<std::endl;
		// Send the local queue pair information to the slave rank
		MPI_Send(&(ibvQpConn_vec[slave_rank]->getQpairStruct()->local), sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0, MPI_COMM_WORLD);
	}
	else if (local_rank == slave_rank)
	{
		std::cout<<"Local rank "<<local_rank<<" receiving remote QP from remote rank "<<master_rank<<std::endl;
		// Receive the queue pair information from the master rank
		fpga::ibvQ received_q;
		MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Copy the received data to the remote queue pair
		ibvQpConn_vec[master_rank]->getQpairStruct()->remote = received_q;
	}

	// Synchronize after the first exchange to avoid race conditions
	MPI_Barrier(MPI_COMM_WORLD);

	if (local_rank == slave_rank)
	{
		std::cout<<"Local rank "<<local_rank<<" sending local QP to remote rank "<<master_rank<<std::endl;
		// Send the local queue pair information to the master rank
		MPI_Send(&(ibvQpConn_vec[master_rank]->getQpairStruct()->local), sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0, MPI_COMM_WORLD);
	}
	else if (local_rank == master_rank)
	{
		std::cout<<"Local rank "<<local_rank<<" receiving remote QP from remote rank "<<slave_rank<<std::endl;
		// Receive the queue pair information from the slave rank
		fpga::ibvQ received_q;
		MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Copy the received data to the remote queue pair
		ibvQpConn_vec[slave_rank]->getQpairStruct()->remote = received_q;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// write established connection to hardware and perform arp lookup
	if (local_rank == master_rank)
	{
		int connection = (ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn & 0xFFFF) | ((ibvQpConn_vec[slave_rank]->getQpairStruct()->remote.qpn & 0xFFFF) << 16);
		ibvQpConn_vec[slave_rank]->getQpairStruct()->print();
		ibvQpConn_vec[slave_rank]->setConnection(connection);
		ibvQpConn_vec[slave_rank]->writeContext(ranks[slave_rank].port);
		ibvQpConn_vec[slave_rank]->doArpLookup();
		ranks[slave_rank].session_id = ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn;
	} else if (local_rank == slave_rank) 
	{
		int connection = (ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn & 0xFFFF) | ((ibvQpConn_vec[master_rank]->getQpairStruct()->remote.qpn & 0xFFFF) << 16);
		ibvQpConn_vec[master_rank]->getQpairStruct()->print();
		ibvQpConn_vec[master_rank]->setConnection(connection);
		ibvQpConn_vec[master_rank]->writeContext(ranks[master_rank].port);
		ibvQpConn_vec[master_rank]->doArpLookup();
		ranks[master_rank].session_id = ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


void configure_cyt_rdma(std::vector<rank_t> &ranks, int local_rank, CoyoteDevice* device)
{

	std::cout<<"Initializing QP connections..."<<std::endl;
	// create queue pair connections
	std::vector<fpga::ibvQpConn*> ibvQpConn_vec;
	// create single page dummy memory space for each qp
	uint32_t n_pages = 1;
	for(int i=0; i<ranks.size(); i++)
	{
		fpga::ibvQpConn* qpConn = new fpga::ibvQpConn(device->coyote_qProc_vec[i], ranks[local_rank].ip, n_pages);
		ibvQpConn_vec.push_back(qpConn);
		// qpConn->getQpairStruct()->print();
	}

	std::cout<<"Exchanging QP..."<<std::endl;
	for(int i=0; i<ranks.size(); i++)
	{
		for(int j=i+1; j<ranks.size();j++)
		{
			exchange_qp(i, j, local_rank, ibvQpConn_vec, ranks);
		}
	}
}

void configure_cyt_tcp(std::vector<rank_t> &ranks, int local_rank, CoyoteDevice* device)
{
	std::cout<<"Configuring Coyote TCP..."<<std::endl;
	// arp lookup
    for(int i=0; i<ranks.size(); i++){
        if(local_rank != i){
            device->get_device()->doArpLookup(_ip_encode(ranks[i].ip));
        }
    }

	//open port 
    for (int i=0; i<ranks.size(); i++)
    {
        uint32_t dstPort = ranks[i].port;
        bool open_port_status = device->get_device()->tcpOpenPort(dstPort);
    }

	std::this_thread::sleep_for(10ms);

	//open con
    for (int i=0; i<ranks.size(); i++)
    {
        uint32_t dstPort = ranks[i].port;
        uint32_t dstIp = _ip_encode(ranks[i].ip);
        uint32_t dstRank = i;
		uint32_t session = 0;
        if (local_rank != dstRank)
        {
            bool success = device->get_device()->tcpOpenCon(dstIp, dstPort, &session);
			ranks[i].session_id = session;
        }
    }

}

#ifdef COYOTE_HSA_SUPPORT

#define FIRST_THREAD_IN_BLOCK() ((threadIdx.x + threadIdx.y + threadIdx.z) == 0)
#define FIRST_BLOCK() ( blockIdx.x + blockIdx.y + blockIdx.z == 0)
#define BEGIN_BLOCK_ZERO_DO __syncthreads(); if(FIRST_THREAD_IN_BLOCK()) { do{
#define END_BLOCK_ZERO_DO }while(0); } __syncthreads();

#define BEGIN_SINGLE_THREAD_DO __threadfence_system(); __syncthreads(); if(FIRST_BLOCK()&&FIRST_THREAD_IN_BLOCK()) { do{
#define END_SINGLE_THREAD_DO }while(0); } __threadfence_system();
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

const std::size_t OFFSET_HOSTCTRL = 0x2000; 
const std::size_t OFFSET_CCLO = 0x0; 

/**
 * Utilities to write and read Coyote registers from GPU kernels
*/
//__device__ inline auto setCSR(volatile uint64_t * ctrl_reg, uint64_t val, uint32_t offs) {printf("write 0x%lx into 0x%lx\n", val, offs); ctrl_reg[offs] = val;}
//__device__ inline auto setCSR(volatile uint64_t * ctrl_reg, uint64_t val, uint32_t offs) {
//	//printf("write 0x%lx into 0x%lx\n", val, offs);
//	//STORE(&ctrl_reg[offs],val);
//	// asm volatile(
//    //      "s_waitcnt lgkmcnt(0)"
//    //  );
//	asm volatile(
//    "flat_store_dwordx2 %0 %1 glc slc \n"
//    :
//    : "v"(&ctrl_reg[offs]), "v"(val));
//    asm volatile(
//        "s_waitcnt lgkmcnt(0)"
//    );
//    asm volatile(
//        "s_waitcnt vmcnt(0)"
//    );
//
//	//asm volatile(
//	//	"flat_store_dwordx2 %0 %1 glc slc \n"
//	//	:
//	//	: "v"(&ctrl_reg[offs]), "v"(val)); // Write first address last
//}
//

__device__ inline auto getCSR(volatile uint64_t * ctrl_reg, uint32_t offs) { return ctrl_reg[offs]; }

__device__ inline auto setCSR(volatile uint64_t * ctrl_reg, uint64_t val, uint32_t offs) {


	int check = 1;
	while(check) {
		int reg = getCSR(ctrl_reg, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
		//check = !((reg >> 1) & 0x1); 
		check = ((reg >> 1) & 0x1); 

	}
	
	asm volatile(
    "flat_store_dwordx2 %0 %1 glc slc \n"
    :
    : "v"(&ctrl_reg[offs]), "v"(val));
    asm volatile(
        "s_waitcnt lgkmcnt(0)"
    );
    asm volatile(
        "s_waitcnt vmcnt(0)"
    );

}



__device__ inline auto start_setCSR(volatile uint64_t * ctrl_reg, uint64_t val, uint32_t offs) {

	int check = 1;
	while(check) {
		int reg = getCSR(ctrl_reg, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
		//check = !((reg >> 1) & 0x1); 
		check = ((reg >> 1) & 0x1); 

	}
	if (!check){
	asm volatile(
    "flat_store_dwordx2 %0 %1 glc slc \n"
    :
    : "v"(&ctrl_reg[offs]), "v"(val));
    asm volatile(
        "s_waitcnt lgkmcnt(0)"
    );
    asm volatile(
        "s_waitcnt vmcnt(0)"
    );
	}

}



/**
 * Utility to create ACCL copy request from GPU kernel
*/

__global__ void starter_ctrl(volatile uint64_t * ctrl_reg){

	//auto start_beforesync = std::chrono::high_resolution_clock::now();
//
	//__threadfence_system(); 
	//__syncthreads();
	//hipStreamSynchronize(0);
	setCSR(ctrl_reg, 0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

	//poll for the end
	int check = 1;
	
	int counter = 0;
	//while(check) {
	//	int reg = getCSR(ctrl_reg, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
	//	check = !((reg >> 1) & 0x1); 
	//	counter++;
	//}


}

__device__ inline void start(::ACCL::GPU::Options options, volatile uint64_t * ctrl_reg) {
        //assert(this->get_status() ==  operationStatus::EXECUTING);

		//volatile int clock_counter = 0;
		//while(clock_counter < 10000000) {
		//	clock_counter++;
		//}
//
		//volatile int clock_counter3 = 0;
		//while(clock_counter3 < 10000000) {
		//	clock_counter3++;
		//}
        int function, arg_id = 0;

        if (options.scenario == ::ACCL::GPU::operation::config) {
            function = static_cast<int>(options.cfg_function);
        } else {
            function = static_cast<int>(options.reduce_function);
        }

        uint32_t flags = static_cast<uint32_t>(options.host_flags) << 8 | static_cast<uint32_t>(options.stream_flags);

		//write parameters
		setCSR(ctrl_reg, static_cast<uint32_t>(options.scenario), (0x2010)>>2);
		printf("Written scenario");
        setCSR(ctrl_reg, static_cast<uint32_t>(options.count), (0x2018)>>2);
        printf("Written count");
		setCSR(ctrl_reg, static_cast<uint32_t>(options.comm), (0x2020)>>2);
        printf("Written comm");
		setCSR(ctrl_reg, static_cast<uint32_t>(options.root_src_dst), (0x2028)>>2);
		printf("Written root_src_dst");
		setCSR(ctrl_reg, static_cast<uint32_t>(function), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::FUNCTION_R)>>2);
        printf("Written function");
		setCSR(ctrl_reg, static_cast<uint32_t>(options.tag), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::MSG_TAG)>>2);
        printf("Written tag");
		setCSR(ctrl_reg, static_cast<uint32_t>(options.arithcfg_addr), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::DATAPATH_CFG)>>2);
        printf("Written compression_flags");
		setCSR(ctrl_reg, static_cast<uint32_t>(options.compression_flags), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMPRESSION_FLAGS)>>2);
        printf("Written stream_flags");
		//hardcoded flags for a generic GPU->FPGA->GPU
		setCSR(ctrl_reg, static_cast<uint32_t>(1280), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::STREAM_FLAGS)>>2);
        addr_t addr_a = options.addr_0;
        printf("Written addr_0");
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_a), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_0)>>2);
        printf("Written addr_1");
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_a >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_1)>>2);
        addr_t addr_b = options.addr_1;
        printf("Written addr_b");
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_b), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_0)>>2);
        printf("Written addr_b");
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_b >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_1)>>2);
        addr_t addr_c = options.addr_2;
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_c), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_0)>>2);
		setCSR(ctrl_reg, static_cast<uint32_t>(addr_c >> 32), (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_1)>>2);

		setCSR(ctrl_reg, 0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);

		//volatile int clock_counter = 0;
		//while(clock_counter < 10000000) {
		//	clock_counter++;
		//}
		//ctrl_reg[(OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2] = 0x1U;


        // //start the kernel
        // setCSR(ctrl_reg, 0x1U, (OFFSET_HOSTCTRL + HOSTCTRL_ADDR::AP_CTRL)>>2);
		
		//kernel completed
		
		//hipLaunchKernelGGL(starter_ctrl,dim3(1),dim3(1),0,stream,(volatile uint64_t *) gpu_ptr); //dynamic parallelism not supported for HIP



    }
/**
 * GPU kernel to launch copy
*/
__global__ void launch_copy( addr_t buffer_a, addr_t buffer_b, uint32_t count, volatile uint64_t * ctrl_reg) {
    // BEGIN_SINGLE_THREAD_DO

        ::ACCL::GPU::Mapping m;
        m.ctrl_reg = (volatile uint64_t *) ctrl_reg;

        //fill parameters

        ::ACCL::GPU::Options options;
        options.scenario = ::ACCL::GPU::operation::copy;
		//printf("addr_0 = %lx, addr_2 = %lx\n", buffer_a, buffer_b);
        options.addr_0 = buffer_a;
        options.addr_2 = buffer_b;
        options.data_type_io_0 = ::ACCL::GPU::dataType::float32;
        options.data_type_io_2 = ::ACCL::GPU::dataType::float32;
        options.count = count;
		options.arithcfg_addr = 196; //both buffers are on the GPU, not on the FPGA
		//printf("count = %d\n", count);
        options.stream_flags = ::ACCL::GPU::streamFlags::NO_STREAM;
        options.waitfor = NULL;

		//printf("scenario = %d, data_type_io_0 = %d, data_type_io_2 = %d, stream_flags = %d\n", options.scenario, options.data_type_io_0, options.data_type_io_2, options.stream_flags);
        
		//create request and launch
		::ACCL::GPU::CoyoteRequest cr{options, &m};
        start(options, ctrl_reg);
		
    //END_SINGLE_THREAD_DO
}


/**
 * @brief exports CTRL registrs via DMABuf, attaches GPU and invokes copy:
 * 
*/
void test_mapping(::ACCL::ACCL &accl, options_t &options) {
	
	unsigned int count = options.nruns;
	
	//allocate GPU buffers

  	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32, true);
  	auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32, true);
	
	//allocate host buffers
  	auto op_buf_host = std::vector<float>(count);
	auto res_buf_host = std::vector<float>(count);
	test_debug("host buffers allocated", options);

	//populate buffers
	random_array(op_buf_host.data(), count);
	hipMemcpy(op_buf->buffer(), op_buf_host.data(), 4*count, hipMemcpyHostToDevice);
	hipMemset(res_buf->buffer(), 78, 4*count);
	
	//get virtual address for FPGA CTRL registers
	auto cclo_ptr = static_cast<CoyoteDevice*>(accl.cclo);
	
	//export this memory area vuia DMABuf
	int dmabuf_fd = (int) cclo_ptr->export_CTRL_registers();
	test_debug("dmabuf fd: " + std::to_string(dmabuf_fd), options);

	//HSA setup
	hsa_agent_t gpu_device;
	hsa_status_t err;
	my_ready = false;
	err = hsa_iterate_agents(find_gpu, &gpu_device);
	assert(err == HSA_STATUS_SUCCESS);
	void * gpu_ptr = NULL;
	size_t gpu_size = 0;
	void * metadata = NULL;
	size_t metadata_size = 0;

	
	//import DMABuf and get GPU pointer
	err = hsa_amd_interop_map_buffer(1, &gpu_device, dmabuf_fd, 0, &gpu_size,  &gpu_ptr, &metadata_size, (const void **) &metadata);
	if(err == HSA_STATUS_SUCCESS) {
		std::cout << "import successful!" << std::endl;
	} else {
		char * err_str;
		hsa_status_string(err, (const char **) &err_str);
		std::cout << "import error: " << err << ": " << err_str << std::endl;
	}
	std::cout << "GPU ptr: " << std::hex << gpu_ptr << std::dec << ", size: " << gpu_size << std::endl;

	
	cclo_ptr->printDebug();

	//input to use to launch chipscope debugging
	int input = 0;
	std::cin >> input;

	//launch GPU kernel
 	hipStream_t stream;
    hipStreamCreate(&stream);
	volatile int clock_counter2 = 0;
	while(clock_counter2 < 10000000) {
		clock_counter2++;
	}
	hipLaunchKernelGGL(launch_copy, 1, 1, 0, stream, (uint64_t) op_buf->buffer(), (uint64_t) res_buf->buffer(), count, (volatile uint64_t *) gpu_ptr);
	hipEvent_t event;
	hipEventCreate(&event);
	hipEventRecord(event, stream);
	volatile int clock_counter = 0;
	while(clock_counter < 10000000) {
		clock_counter++;
	}
	// auto start = std::chrono::high_resolution_clock::now();
	//hipLaunchKernelGGL(starter_ctrl,dim3(1),dim3(1),0,stream,(volatile uint64_t *) gpu_ptr);
	hipStreamWaitEvent(stream, event, 0);
	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsedTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
	//ofstream outputFile;
	//outputFile.open("SecondCall.csv", std::ios_base::app);
	//outputFile<<elapsedTime<<std::endl;
	//outputFile.close();
	
	//synchronize GPU kernel 

	hipDeviceSynchronize();
	//int input2 = 0;
	//std::cout<<"Insert value to continue - WARNING: usually, crash after this step occurs " << std::endl;
	//std::cin >> input2;


	
	cclo_ptr->printDebug(); //see how many writes have been performed w.r.t. previous printDebug() invokation

	//current status of the registers
	std::cout << "Currently reading FPGA registers to check the status" << std::endl;
	std::cout << "scenario: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::SCEN)>>2) << std::endl
	<< "count: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::LEN)>>2) <<std::endl
	<< "comm: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMM)>>2)<<std::endl
	<< "root_src_dst: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ROOT_SRC_DST)>>2)  <<std::endl
	<< "function: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::FUNCTION_R)>>2)<<std::endl
	<< "tag: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::MSG_TAG)>>2) <<std::endl
	<< "arithcfg_addr: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::DATAPATH_CFG)>>2)<<std::endl
	<< "compression flags: " <<  cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::COMPRESSION_FLAGS)>>2)<<std::endl
	<< "stream flags: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::STREAM_FLAGS)>>2)<<std::endl
	<< "addr_a0: " << std::hex << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_0)>>2)<<std::endl
	<< "addr_a1: " << std::hex << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRA_1)>>2)<<std::endl
	<< "addr b0: " << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_0)>>2)<<std::endl
	<< "addr_b1: " << std::hex << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRB_1)>>2)<<std::endl
	<< "addr_c0: " << std::hex << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_0)>>2)<<std::endl
	<< "addr_c1: " << std::hex << cclo_ptr->coyote_proc->getCSR((OFFSET_HOSTCTRL + HOSTCTRL_ADDR::ADDRC_1)>>2) << std::endl;
	std::cout << "Status checking completed" << std::endl;

	std::cout << "duration from CPU: " << std::hex << cclo_ptr->read(CCLO_ADDR::PERFCNT_OFFSET) << std::dec << std::endl;
	std::cout << "retcode: " << std::hex << cclo_ptr->read(CCLO_ADDR::RETCODE_OFFSET) << std::dec << std::endl;
	
	//verify correctness of the copy
	hipMemcpy(res_buf_host.data(), res_buf->buffer(), 4*count, hipMemcpyDeviceToHost);
	int check = memcmp(op_buf_host.data(), res_buf_host.data(), sizeof(float) * count);

	int pos = 0;

	if(!check) {
		std::cout << "Copy is successful!" << std::endl;
	} else {
		std::cout << "Test is NOT successful!" << std::endl;
	}
}

#endif

#ifdef COYOTE_HSA_SUPPORT
/**
 * @brief performs three different types of copies between GPU buffers and the FPGA, with peer-to-peer DMA:
 * 1. GPU buffer #1 -> FPGA -> GPU buffer #2
 * 2. FPGA -> GPU buffer #2
 * 3. GPU buffer #1 -> FPGA
 * 
 * @returns HW times (microseconds) averaged over 100 iterations 
*/
void test_copy_averaged_gpu_p2p(::ACCL::ACCL &accl, options_t &options) {
	ofstream outputFile;
	outputFile.open(options.output_file, std::ios_base::app);

	unsigned int count = options.nruns;

	//create two buffers on the GPU
	
  	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32, true);
  	auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32, true);
	
	//create two buffers on the host
  	auto op_buf_host = std::vector<float>(count);
	auto res_buf_host = std::vector<float>(count);
	test_debug("HOST and GPU buffers allocated", options);

	//create vectors to save times and compute averages
	std::vector<float> copy_values, sync_from_values, sync_to_values;

	//initialize first GPU buffer with random data
	random_array(op_buf_host.data(), count);
	hipMemcpy(op_buf->buffer(), op_buf_host.data(), 4*count, hipMemcpyHostToDevice);

	//initialize second GPU buffer with fixed data
	hipMemset(res_buf->buffer(), 78, 4*count);

	test_debug("copy type 1: GPU buffer #1 -> FPGA -> GPU buffer #2", options);

	double sw_time_us = 0.0, hw_time_us = 0.0;

	for(int i = 0; i < 10; i++) {	
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 1 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		copy_values.push_back(hw_time_us);
	}

	//check correctness of the copy
	hipMemcpy(res_buf_host.data(), res_buf->buffer(), 4*count, hipMemcpyDeviceToHost);
	int check = memcmp(op_buf_host.data(), res_buf_host.data(), sizeof(float) * count);	

	if(!check) {
		std::cout << "Copy is successful!" << std::endl;
	} else {
		std::cout << "Copy is NOT successful!" << std::endl;
	}

	test_debug("copy type 2: FPGA -> GPU buffer #2", options);

	for(int i = 0; i < 10; i++) {	
		op_buf->sync_to_device();
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 2 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_from_values.push_back(hw_time_us);
	}

	test_debug("copy type 3: GPU buffer #1 -> FPGA", options);

	for(int i = 0; i < 10; i++) {
		res_buf->sync_to_device();
		op_buf->sync_from_device();
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 3 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_to_values.push_back(hw_time_us);
	}

	//compute averages

	auto avg_copy = std::reduce(std::begin(copy_values), std::end(copy_values)) / copy_values.size();
	
	std::cout << "Average HW time for type #1 (GPU buffer #1 -> FPGA -> GPU buffer #2): " << avg_copy << std::endl;

	auto avg_sync_to = std::reduce(std::begin(sync_to_values), std::end(sync_to_values)) / sync_to_values.size();

	std::cout << "Average HW time for type #2 (FPGA -> GPU buffer #2): " << avg_sync_to << std::endl;

	auto avg_sync_from = std::reduce(std::begin(sync_from_values), std::end(sync_from_values)) / sync_from_values.size();
	
	std::cout << "Average HW time for type #3 (GPU buffer #1 -> FPGA): " << avg_sync_from << std::endl;

	outputFile << count << ", " << avg_copy << ", " << avg_sync_from << ", " << avg_sync_to << std::endl;

	outputFile.close();
	
	//release memory 

	op_buf->free_buffer();
	res_buf->free_buffer();
}

/**
 * @brief performs three different types of copies between GPU buffers and the FPGA, without peer-to-peer DMA:
 * 1. GPU buffer #1 -> HOST buffer #1 -> FPGA -> HOST buffer #2 -> GPU buffer #2
 * 2. FPGA -> HOST buffer #2 -> GPU buffer #2
 * 3. GPU buffer #1 -> HOST buffer #1 -> FPGA
 * 
 * @returns SW times (microseconds) averaged over 100 iterations 
*/
void test_copy_averaged_gpu_no_p2p(::ACCL::ACCL &accl, options_t &options) {
	ofstream outputFile;
	outputFile.open(options.output_file, std::ios_base::app);
	unsigned int count = options.nruns;

	//create two buffers on the HOST
  	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32, false);
  	auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32, false);
	
	//create two buffers on the GPU
	hsa_agent_t gpu_device;
	hsa_status_t err = hsa_iterate_agents(find_gpu, &gpu_device);
	if(err != HSA_STATUS_SUCCESS) {
	throw std::runtime_error("No GPU found!");
	}

	hsa_region_t region_to_use = {0}; 
	struct get_region_info_params info_params = {
		.region = &region_to_use,
		.desired_allocation_size = count * sizeof(float)
	};
	hsa_agent_iterate_regions(gpu_device, get_region_info, &info_params);
	err = (region_to_use.handle == 0) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
	if(err != HSA_STATUS_SUCCESS) {
	throw std::runtime_error("Insufficient memory on the GPU!");
	}

	void * op_buf_gpu;
	void * res_buf_gpu;

	err = hsa_memory_allocate(region_to_use, sizeof(float) * count, (void **) &(op_buf_gpu));
	if(err != HSA_STATUS_SUCCESS) {
	throw std::runtime_error("Allocation failed on the GPU!");
	}
	err = hsa_memory_allocate(region_to_use, sizeof(float) * count, (void **) &(res_buf_gpu));
	if(err != HSA_STATUS_SUCCESS) {
	throw std::runtime_error("Allocation failed on the GPU!");
	}

	test_debug("HOST and GPU buffers allocated", options);	

	//allocate buffer on the host
	auto op_buf_data = std::vector<float>(count);
	auto res_buf_data = std::vector<float>(count);

	//initialize HOST buffer #1 and GPU buffer #1 with random data
	random_array(op_buf_data.data(), count);
	hipMemcpy(op_buf_gpu, op_buf_data.data(), 4*count, hipMemcpyHostToDevice);

	//initialize GPU buffer #2 with fixed data
	hipMemset(res_buf_gpu, 78, 4*count);
	
	//create vectors to save times and compute averages
	std::vector<float> copy_values, sync_from_values, sync_to_values;

	test_debug("copy type 1: GPU buffer #1 -> HOST buffer #1 -> FPGA -> HOST buffer #2 -> GPU buffer #2", options);

	double sw_time_us = 0.0, hw_time_us = 0.0;

	for(int i = 0; i < 100; i++) {	
		auto start = std::chrono::high_resolution_clock::now();
		hipMemcpy(op_buf->buffer(), op_buf_gpu, 4*count, hipMemcpyDeviceToHost);
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		hipMemcpy(res_buf_gpu, res_buf->buffer(), 4*count, hipMemcpyHostToDevice);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 1 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		copy_values.push_back(sw_time_us);
	}

	//check correctness of the copy
	hipMemcpy(res_buf->buffer(), res_buf_gpu, 4*count, hipMemcpyDeviceToHost);
	int check = memcmp(op_buf->buffer(), res_buf->buffer(), sizeof(float) * count);	

	if(!check) {
		std::cout << "Copy is successful!" << std::endl;
	} else {
		std::cout << "Copy is NOT successful!" << std::endl;
	}

	test_debug("copy type 2: FPGA -> HOST buffer #2 -> GPU buffer #2", options);

	for(int i = 0; i < 100; i++) {	
		op_buf->sync_to_device();
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		hipMemcpy(res_buf_gpu, res_buf->buffer(), 4*count, hipMemcpyHostToDevice);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 2 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_from_values.push_back(sw_time_us);
	}

	test_debug("copy type 3: GPU buffer #1 -> HOST buffer #1 -> FPGA", options);

	for(int i = 0; i < 100; i++) {
		res_buf->sync_to_device();
		op_buf->sync_from_device();
		auto start = std::chrono::high_resolution_clock::now();
		hipMemcpy(op_buf->buffer(), op_buf_gpu, 4*count, hipMemcpyDeviceToHost);
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 3 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_to_values.push_back(sw_time_us);
	}

	//compute averages

	auto avg_copy = std::reduce(std::begin(copy_values), std::end(copy_values)) / copy_values.size();
	
	std::cout << "Average SW time for type #1 (GPU buffer #1 -> HOST buffer #1 -> FPGA -> HOST buffer #2 -> GPU buffer #2): " << avg_copy << std::endl;

	auto avg_sync_to = std::reduce(std::begin(sync_to_values), std::end(sync_to_values)) / sync_to_values.size();

	std::cout << "Average SW time for type #2 (FPGA -> HOST buffer #2 -> GPU buffer #2): " << avg_sync_to << std::endl;

	auto avg_sync_from = std::reduce(std::begin(sync_from_values), std::end(sync_from_values)) / sync_from_values.size();
	
	std::cout << "Average SW time for type #3 (GPU buffer #1 -> HOST buffer #1 -> FPGA): " << avg_sync_from << std::endl;

	outputFile << count << ", " << avg_copy << ", " << avg_sync_from << ", " << avg_sync_to << std::endl;

	outputFile.close();
	
	//release memory 

	op_buf->free_buffer();
	res_buf->free_buffer();

	hsa_memory_free(op_buf_gpu);
	hsa_memory_free(res_buf_gpu);
}
#endif

/**
 * @brief performs three different types of copies between FPGA buffers and the HOST:
 * 1. HOST buffer #1 -> FPGA -> HOST buffer #2
 * 2. FPGA -> HOST buffer #2
 * 3. HOST buffer #1 -> FPGA
 * 
 * @returns HW times (microseconds) averaged over 100 iterations 
*/
void test_copy_averaged_host(::ACCL::ACCL &accl, options_t &options) {
	ofstream outputFile;
	outputFile.open(options.output_file, std::ios_base::app);
	unsigned int count = options.nruns;

	//create two buffers on the host
  	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
  	auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	
	test_debug("HOST buffers allocated", options);

	//create vectors to save times and compute averages
	std::vector<float> copy_values, sync_from_values, sync_to_values;

	//initialize first HOST buffer with random data
	random_array(op_buf->buffer(), count);

	//initialize second HOST buffer with fixed data
	memset(res_buf->buffer(), 78, 4*count);

	test_debug("copy type 1: HOST buffer #1 -> FPGA -> HOST buffer #2", options);

	double sw_time_us = 0.0, hw_time_us = 0.0;

	for(int i = 0; i < 100; i++) {	
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 1 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		copy_values.push_back(hw_time_us);
	}

	//check correctness of the copy
	int check = memcmp(op_buf->buffer(), res_buf->buffer(), sizeof(float) * count);	

	if(!check) {
		std::cout << "Copy is successful!" << std::endl;
	} else {
		std::cout << "Copy is NOT successful!" << std::endl;
	}

	test_debug("copy type 2: FPGA -> HOST buffer #2", options);

	for(int i = 0; i < 100; i++) {	
		op_buf->sync_to_device();
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 2 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_from_values.push_back(hw_time_us);
	}

	test_debug("copy type 3: HOST buffer #1 -> FPGA", options);

	for(int i = 0; i < 100; i++) {
		res_buf->sync_to_device();
		op_buf->sync_from_device();
		auto start = std::chrono::high_resolution_clock::now();
		auto req = accl.copy(*op_buf, *res_buf, count, true, true, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		sw_time_us = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		hw_time_us = accl.get_duration(req) / 1000.0;
		test_debug("copy type 3 #" + std::to_string(i) + ": sw_time = " + std::to_string(sw_time_us) + " us, hw_time = " + std::to_string(hw_time_us) + " us", options);
		sync_to_values.push_back(hw_time_us);
	}

	//compute averages

	auto avg_copy = std::reduce(std::begin(copy_values), std::end(copy_values)) / copy_values.size();
	
	std::cout << "Average HW time for type #1 (HOST buffer #1 -> FPGA -> HOST buffer #2): " << avg_copy << std::endl;

	auto avg_sync_to = std::reduce(std::begin(sync_to_values), std::end(sync_to_values)) / sync_to_values.size();

	std::cout << "Average HW time for type #2 (FPGA -> HOST buffer #2): " << avg_sync_to << std::endl;

	auto avg_sync_from = std::reduce(std::begin(sync_from_values), std::end(sync_from_values)) / sync_from_values.size();
	
	std::cout << "Average HW time for type #3 (HOST buffer #1 -> FPGA): " << avg_sync_from << std::endl;

	outputFile << count << ", " << avg_copy << ", " << avg_sync_from << ", " << avg_sync_to << std::endl;

	outputFile.close();
	
	//release memory 

	op_buf->free_buffer();
	res_buf->free_buffer();
}

void test_sendrcv(::ACCL::ACCL &accl, options_t &options) {
  	std::cout << "Start send recv test..." << std::endl<<std::flush;
	// do the send recv test here
	int bufsize = options.count;

	if (options.count*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<int>(bufsize, dataType::int32);
	
	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;

		// rank 0 initializes the buffer with numbers, rank1 with -1
		for (int i = 0; i < bufsize; i++) op_buf.get()->buffer()[i] = (mpi_rank == 0) ? i : -1;
		
		if (options.host == 0){ op_buf->sync_to_device(); }

		MPI_Barrier(MPI_COMM_WORLD);
		
		double durationUs = 0.0;
		double tput = 0.0;
		auto start = std::chrono::high_resolution_clock::now();

		::ACCL::ACCLRequest* req;
		if (mpi_rank == 0) {
			// send
			req = accl.send(*op_buf, bufsize, 1, TAG_ANY, GLOBAL_COMM, true, dataType::none, true); // most default send from 0 to 1
			accl.wait(req, 1000ms);

		} else if (mpi_rank == 1) {
			// receive
			req = accl.recv(*op_buf, bufsize, 0, TAG_ANY, GLOBAL_COMM, true, dataType::none, true); // most default recv to 1 from 0
			accl.wait(req, 1000ms);
		}
		
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		tput = (options.count*sizeof(dataType::int32)*8.0)/(durationUs*1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;
		std::cout<<"host measured tput:"<<tput<<std::endl;

		std::cerr << "Rank " << mpi_rank << " passed barrier after send recv test!" << std::endl;
		
		if (mpi_rank == 0 || mpi_rank == 1){
			durationUs = (double)accl.get_duration(req)/1000.0;
			tput = (options.count*sizeof(dataType::int32)*8.0)/(durationUs*1000.0);
			if(durationUs > 1.0){
				accl_log(mpi_rank, format_log("sendrecv", options, durationUs, tput));
			}
		}

		int errors = 0;

		if (options.host == 0){ op_buf->sync_from_device(); }
		
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
	
	op_buf->free_buffer();
}


void test_bcast(::ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start bcast test with root " + std::to_string(root) + " ..."
				<< std::endl<<std::flush;
	MPI_Barrier(MPI_COMM_WORLD);
	unsigned int count = options.count;

	if (options.count*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<int>(count, dataType::int32);
	
	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;
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
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.bcast(*op_buf, count, root, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;

		std::this_thread::sleep_for(10ms);

		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("bcast", options, durationUs, 0));
		}

		if (options.host == 0){ op_buf->sync_from_device(); }

		if (mpi_rank != root) {
			int errors = 0;
			for (int i = 0; i < count; i++) {
				unsigned int res = op_buf.get()->buffer()[i];
				unsigned int ref = i;
				if (res != ref) {
					std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
									std::to_string(res) + " != " + std::to_string(ref) +
									")"
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
	}

	op_buf->free_buffer();

}

void test_scatter(::ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start scatter test with root " + std::to_string(root) + " ..."
				<< std::endl;
	unsigned int count = options.count;

	if (options.count*mpi_size*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<int>(count * mpi_size, dataType::int32);
	auto res_buf = accl.create_coyotebuffer<int>(count, dataType::int32);

	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;

		// op buf initialized with i and res buf initialized with -1
		for (int i = 0; i < count * mpi_size; i++) op_buf.get()->buffer()[i] = i;
		for (int i = 0; i < count; i++) res_buf.get()->buffer()[i] = -1;

		if (options.host == 0){ op_buf->sync_to_device(); }
		if (options.host == 0){ res_buf->sync_to_device(); }

		test_debug("Scatter data from " + std::to_string(root) + "...", options);
		
		MPI_Barrier(MPI_COMM_WORLD);
		double durationUs = 0.0;
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;

		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("scatter", options, durationUs, 0));
		}

		std::this_thread::sleep_for(10ms);
		if (options.host == 0){ op_buf->sync_from_device(); }
		if (options.host == 0){ res_buf->sync_from_device(); }

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

	op_buf->free_buffer();
	res_buf->free_buffer();

}


void test_gather(::ACCL::ACCL &accl, options_t &options, int root) {
	std::cout << "Start gather test with root " + std::to_string(root) + "..."
				<< std::endl;
	unsigned int count = options.count;

	if (options.count*mpi_size*sizeof(dataType::float32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	
	std::unique_ptr<::ACCL::Buffer<float>> res_buf;
	if (mpi_rank == root) {
		res_buf = accl.create_coyotebuffer<float>(count * mpi_size, dataType::float32);
	} else {
		res_buf = std::unique_ptr<::ACCL::Buffer<float>>(nullptr);
	}

	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;


		for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = mpi_rank*count + i;
		if (options.host == 0){ op_buf->sync_to_device(); }
		if (mpi_rank == root) {
			for (int i = 0; i < count * mpi_size; i++) res_buf.get()->buffer()[i] = 0;
			if (options.host == 0){ res_buf->sync_to_device(); }
		}

		test_debug("Gather data from " + std::to_string(mpi_rank) + "...", options);

		MPI_Barrier(MPI_COMM_WORLD);
		double durationUs = 0.0;
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);

		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;
		
		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("gather", options, durationUs, 0));
		}

		std::this_thread::sleep_for(10ms);
		if (options.host == 0){ op_buf->sync_from_device(); }
		if (mpi_rank == root){
			if (options.host == 0){ res_buf->sync_from_device(); }
		}

		if (mpi_rank == root) {
			int errors = 0;
			for (unsigned int j = 0; j < mpi_size; ++j) {
				for (size_t i = 0; i < count; i++)
				{
					float res = res_buf.get()->buffer()[j*count+i];
					float ref = j*count+i;
					if (res != ref) {
						std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
										std::to_string(res) + " != " + std::to_string(ref) +
										")"
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
	}

	op_buf->free_buffer();
	if (mpi_rank == root) {
		res_buf->free_buffer();
	}
}


void test_allgather(::ACCL::ACCL &accl, options_t &options) {
	std::cout << "Start allgather test..." << std::endl;
	unsigned int count = options.count;

	if (options.count*mpi_size*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	auto res_buf = accl.create_coyotebuffer<float>(count * mpi_size, dataType::float32);
	
	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;

		for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = mpi_rank*count + i;
		for (int i = 0; i < count * mpi_size; i++) res_buf.get()->buffer()[i] = 0;

		if (options.host == 0){ op_buf->sync_to_device(); }
		if (options.host == 0){ res_buf->sync_to_device(); }

		test_debug("Gathering data...", options);

		MPI_Barrier(MPI_COMM_WORLD);
		double durationUs = 0.0;
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.allgather(*op_buf, *res_buf, count, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;

		

		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("allgather", options, durationUs, 0));
		}

		std::this_thread::sleep_for(10ms);

		if (options.host == 0){ op_buf->sync_from_device(); }
		if (options.host == 0){ res_buf->sync_from_device(); }

		int errors = 0;
		for (unsigned int j = 0; j < mpi_size; ++j) {
			for (size_t i = 0; i < count; i++)
			{
				float res = res_buf.get()->buffer()[j*count+i];
				float ref = j*count+i;
				if (res != ref) {
					std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
									std::to_string(res) + " != " + std::to_string(ref) +
									")"
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

	op_buf->free_buffer();
	res_buf->free_buffer();

}

void test_reduce(::ACCL::ACCL &accl, options_t &options, int root,
                 reduceFunction function) {
	std::cout << "Start reduce test with root " + std::to_string(root) +
					" and reduce function " +
					std::to_string(static_cast<int>(function)) + "..."
				<< std::endl;
	unsigned int count = options.count;

	if (options.count*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<float>(count, dataType::float32);
	auto res_buf = accl.create_coyotebuffer<float>(count, dataType::float32);

	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;
		for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = i;
		for (int i = 0; i < count; i++) res_buf.get()->buffer()[i] = 0;

		if (options.host == 0){ op_buf->sync_to_device(); }
		if (options.host == 0){ res_buf->sync_to_device(); }

		test_debug("Reduce data to " + std::to_string(root) + "...", options);

		MPI_Barrier(MPI_COMM_WORLD);
		double durationUs = 0.0;
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;

		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("reduce", options, durationUs, 0));
		}

		std::this_thread::sleep_for(10ms);
		if (options.host == 0){ op_buf->sync_from_device(); }
		if (options.host == 0){ res_buf->sync_from_device(); }

		if (mpi_rank == root) {
			int errors = 0;

			for (unsigned int i = 0; i < count; ++i) {
			float res = res_buf.get()->buffer()[i];
			float ref = i * mpi_size;

			if (res != ref) {
				std::cout << std::to_string(i + 1) + "th item is incorrect! (" +
				                 std::to_string(res) + " != " + std::to_string(ref) +
				                 ")"
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
	}

  	op_buf->free_buffer();
	res_buf->free_buffer();
}

void test_allreduce(::ACCL::ACCL &accl, options_t &options,
                    reduceFunction function) {
	std::cout << "Start allreduce test and reduce function " +
					std::to_string(static_cast<int>(function)) + "..."
				<< std::endl;
	unsigned int count = options.count;

	if (options.count*sizeof(dataType::int32) > options.rxbuf_size){
		std::cout<<"experiment size larger than buffer size, exiting..."<<std::endl;
		return;
	}

	auto op_buf = accl.create_coyotebuffer<int>(count, dataType::int32);
	auto res_buf = accl.create_coyotebuffer<int>(count, dataType::int32);

	for (int n = 0; n < options.nruns; n++)
	{
		std::cout << "Repetition " <<n<< std::endl<<std::flush;
		for (int i = 0; i < count; i++) op_buf.get()->buffer()[i] = i;
		for (int i = 0; i < count; i++) res_buf.get()->buffer()[i] = 0;

		if (options.host == 0){ op_buf->sync_to_device(); }
		if (options.host == 0){ res_buf->sync_to_device(); }

		test_debug("Reducing data...", options);

		MPI_Barrier(MPI_COMM_WORLD);
		double durationUs = 0.0;
		accl.barrier();
		std::cout<<"Pass accl barrier"<<std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		::ACCL::ACCLRequest* req = accl.allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, true, true, dataType::none, true);
		accl.wait(req, 1000ms);
		auto end = std::chrono::high_resolution_clock::now();
		durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
		std::cout<<"host measured durationUs:"<<durationUs<<std::endl;

		durationUs = (double)accl.get_duration(req)/1000.0;
		if(durationUs > 1.0){
			accl_log(mpi_rank, format_log("allreduce", options, durationUs, 0));
		}

		std::this_thread::sleep_for(10ms);
		if (options.host == 0){ op_buf->sync_from_device(); }
		if (options.host == 0){ res_buf->sync_from_device(); }

		int errors = 0;

		for (unsigned int i = 0; i < count; ++i) {
			float res = res_buf.get()->buffer()[i];
			float ref = i * mpi_size;

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

	op_buf->free_buffer();
	res_buf->free_buffer();

}

std::unique_ptr<::ACCL::ACCL> accl;

void test_accl_base(options_t options)
{
	std::cout << "Testing ACCL base functionality..." << std::endl;

	// initialize ACCL
	std::vector<::ACCL::rank_t> ranks;
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

		if(options.hardware && options.rdma) {
			rank_t new_rank = {ip, options.start_port, i, options.rxbuf_size};
			ranks.emplace_back(new_rank);
		} else {
			rank_t new_rank = {ip, options.start_port + i, 0, options.rxbuf_size};
			ranks.emplace_back(new_rank);
		}
		
	}
	
	std::unique_ptr<::ACCL::ACCL> accl;
	// construct CoyoteDevice out here already, since it is necessary for creating buffers
	// before the ACCL instance exists.
	::ACCL::CoyoteDevice* device;
	
	MPI_Barrier(MPI_COMM_WORLD);

	if (options.tcp){
		device = new CoyoteDevice();
		configure_cyt_tcp(ranks, local_rank, device);
	} else if (options.rdma){
		device = new CoyoteDevice(mpi_size);
		configure_cyt_rdma(ranks, local_rank, device);
	} else {
		device = new CoyoteDevice();
	}
	
	
	if (options.hardware)
	{
		if (options.udp)
		{
			debug("ERROR: we don't support UDP for now!!!");
			exit(1);
		}
		else if (options.tcp || options.rdma)
		{
			uint localFPGAIP = _ip_encode(ipList[mpi_rank]);
			std::cout << "rank: " << mpi_rank << " FPGA IP: " << std::hex << localFPGAIP << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		accl = std::make_unique<::ACCL::ACCL>(device);
		if (options.protoc == 0){
			std::cout<<"Eager Protocol"<<std::endl;
			accl.get()->initialize(ranks, mpi_rank,
				mpi_size+2, options.rxbuf_size, options.seg_size, 4096*1024*2);
		} else if (options.protoc == 1){
			std::cout<<"Rendezvous Protocol"<<std::endl;
			accl.get()->initialize(ranks, mpi_rank, mpi_size, 64, 64, options.seg_size);
		}  

		debug(accl->dump_communicator());

		MPI_Barrier(MPI_COMM_WORLD);

	} else {
		debug("unsupported situation!!!");
		exit(1);
	}

	
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	double durationUs = 0.0;
	auto start = std::chrono::high_resolution_clock::now();
	::ACCL::ACCLRequest* req = accl->nop(true);
  	accl->wait(req);
	auto end = std::chrono::high_resolution_clock::now();
	durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
	uint64_t durationNs = accl->get_duration(req);
	std::cout << "sw nop time [us]:"<<durationUs<< std::endl;
	std::cout << "hw nop time [ns]:"<< std::dec<< durationNs<< std::endl;

	std::cerr << "Rank " << mpi_rank << " passed last barrier before test!" << std::endl << std::flush;

	MPI_Barrier(MPI_COMM_WORLD);

	
	#ifdef COYOTE_HSA_SUPPORT
	if(options.test_mode == ACCL_COPY || options.test_mode == 0) {
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_copy_averaged_gpu_p2p(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_MAPPING || options.test_mode == 0) {
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_mapping(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	#endif
	if(options.test_mode == ACCL_COPY_FPGA || options.test_mode == 0) {
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_copy_averaged_host(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	
	#ifdef COYOTE_HSA_SUPPORT
	if(options.test_mode == ACCL_COPY_NODMA || options.test_mode == 0) {
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_copy_averaged_gpu_no_p2p(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	#endif
	if(options.test_mode == ACCL_SEND || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_sendrcv(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_BCAST || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_bcast(*accl, options, 0);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_SCATTER || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_scatter(*accl, options, 0);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_GATHER || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_gather(*accl, options, 0);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_ALLGATHER || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_allgather(*accl, options);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_REDUCE || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		int root = 0;
		test_reduce(*accl, options, root, reduceFunction::SUM);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_ALLREDUCE || options.test_mode == 0){
		debug(accl->dump_eager_rx_buffers(false));
		MPI_Barrier(MPI_COMM_WORLD);
		test_allreduce(*accl, options, reduceFunction::SUM);
		debug(accl->dump_communicator());
		debug(accl->dump_eager_rx_buffers(false));
	}
	if(options.test_mode == ACCL_BARRIER){
		std::cout << "Start barrier test..."<< std::endl;
		for (int n = 0; n < options.nruns; n++)
		{
			std::cout << "Repetition " <<n<< std::endl<<std::flush;

			MPI_Barrier(MPI_COMM_WORLD);
			double durationUs = 0.0;
			auto start = std::chrono::high_resolution_clock::now();
			accl->barrier();
			auto end = std::chrono::high_resolution_clock::now();
			durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
			std::cout<<"barrier durationUs:"<<durationUs<<std::endl;
		}
	}

	
	if (failed_tests == 0){
		std::cout << "\nACCL base functionality test completed successfully!\n" << std::endl;
	}
	else {
		std::cout << "\nERROR: ACCL base functionality test failed!\n" << std::endl;
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

void accl_sa_handler(int)
{
	static bool once = true;
	if(once) {
		accl.reset();
		std::cout << "Error! Signal received. Finalizing MPI..." << std::endl;
		MPI_Finalize();
		std::cout << "Done. Terminating..." << std::endl;
		once = false;
	}
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = accl_sa_handler;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT,&sa,NULL);
	sigaction(SIGSEGV, &sa, NULL);
	sigaction(SIGABRT, &sa, NULL);
	std::cout << "Arguments: ";
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
