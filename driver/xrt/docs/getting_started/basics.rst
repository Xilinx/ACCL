..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

##################################
Basics of using ACCL
##################################

Setting up ACCL
****************************

The ACCL driver enables connecting to an Alveo board or to the ACCL simulator or emulator.
The following code snippet illustrates the process of creating an instance of the  ACCL driver in both of these scenarios.

.. code-block:: c++

   std::vector<rank_t> ranks = {};
   for (int i = 0; i < nranks; ++i) {
      rank_t new_rank = {ip[i], port[i], i, rxbuf_size};
      ranks.emplace_back(new_rank);
   }

   if(!use_simulator){
      auto device = xrt::device(0);
      auto xclbin_uuid = device.load_xclbin("example.xclbin");
      auto cclo_ip = xrt::ip(device, xclbin_uuid, "ccl_offload:{ccl_offload_0}");
      auto hostctrl_ip = xrt::kernel(device, xclbin_uuid, "hostctrl:{hostctrl_0}", xrt::kernel::cu_access_mode::exclusive);

      // Configure POE here

      accl = std::make_unique<ACCL::ACCL>(
         ranks, rank, device, cclo_ip, hostctrl_ip, devicemem, rxbufmem,
         networkmem, use_udp ? networkProtocol::UDP : networkProtocol::TCP,
         rxbuf_count, rxbuf_size
      );
   } else {
      accl = std::make_unique<ACCL::ACCL>(
         ranks, rank, sim_start_port, device,
         use_udp ? networkProtocol::UDP : networkProtocol::TCP,
         rxbuf_count, rxbuf_size
      );
   }

Note that when we're executing on Alveo, we must pass into the ACCL constructor XRT handles to the 
CCLO IP and host controller kernel, which themselves are obtained with reference to an XCLBIN ID programmed 
into the Alveo device we're targeting, through the XRT API. 
The `devicemem`, `networkmem`, and `rxbufmem` are integer memory bank identifiers and point to memories used 
for user, POE, and RX buffers respectively.
Not illustrated is any sequence required to configure the POE of choice.
Conversely, when executing against the simulator or emulator, we only need to pass the base port of the emulator 
or simulator process, typically 5500.

Allocating buffers
****************************

For ease-of-use, the ACCL driver provides a dedicated buffer data structure and associated allocation functions.
ACCL buffers allocated through these functions always reside in the user memory bank defined by the bank ID 
passed into the ACCL constructor. See small example below and the API reference for more details.

.. code-block:: c++

   //allocate ACCL buffer of floats
   auto new_buf = accl.create_buffer<float>(count, dataType::float32);
   //wrap an existing XRT BO and expose as ACCL buffer of floats
   auto wrapped_bo = accl.create_buffer<float>(existing_bo, count, dataType::float32);

The ACCL buffer packages datatype-related metadata with the actual data buffer, which if needed is re-aligned to 4K boundaries
as per XRT requirements to enable its synchronization to FPGA memory. 
ACCL buffers also make it transparent to the user whether the buffer lives in actual FPGA memory, or in simulator/emulator memory.

Running ACCL primitives and collectives
*******************************************

ACCL primitives are a set of simple operations that an ACCL instance can execute and assemble into larger operations 
such as collectives. The primitives are:

* Copy - a simple DMA operation from a local source buffer to a local destination buffer
* Combine - applying a binary elementwise operator to two source buffers and placing the result in the destination buffer
* Send - send data from a local buffer to a remote ACCL instance (equivalent to MPI Send)
* Put - send data from a local buffer or stream to a remote stream (roughly equivalent to MPI Put, but result is not placed in memory)
* Receive - receive data from a remote ACCL instance into a local buffer (equivalent to MPI Recv)

ACCL collectives are MPI-like communication primitives assembled from primitives. 
Here is a toy example of an allreduce operation on single precision floating point buffers, 
where `count` denotes the number of FP32 elements in the buffer, not the number of bytes:

.. code-block:: c++

   auto op_buf = accl.create_buffer<float>(count, dataType::float32);
   auto res_buf = accl.create_buffer<float>(count, dataType::float32);
   accl.allreduce(*op_buf, *res_buf, count, reduceFunction::SUM);

Performance optimization
****************************

There are several factors influencing the duration of an ACCL API call:

* the complexity of a call - a copy will be faster than an all-reduce for example
* the size (in bytes) of communicated buffers and their location in the memory hierarchy
* memory contention between sending and receiving processes. ACCL can be configured in specific ways to minimize this contention
* use of blocking or non-blocking variants of the API calls
* network performance, which in itself might depend on the size of buffers i.e. very small buffers typically lead to low utilization of Ethernet bandwidth

Factors which should not influence runtime are:

* data type - API calls on buffers of the same byte size should take the same amount of time, even if the buffers themselves differ in datatype and number of elements
* use of compression - ACCL is designed to perform compression at network rate

The first performance optimization should always be to minimize data movement.
Every ACCL primitive or collective assumes your source and destination buffers are in host memory. 
As such, before the operation is initiated, the source data is moved to the FPGA device memory, and after it completes, 
the resulting data is moved back to host memory. 
These copies have a performance overhead which typically depends on the size of copied buffers.
Programmers can set the `from_fpga` and `to_fpga` optional arguments on most PyACCL calls to indicate 
to the ACCL driver where a synchronization is not necessary.

Secondly, programmers should strive to execute ACCL calls in parallel to other useful CPU work whenever possible,
either by launching ACCL calls in a separate thread, or by using the `async` optional argument that most ACCL calls take. 
If this is set to true, the ACCL function call immediately returns a handle which can be waited on to determine if the processing has actually finished. 
This enables the program to continue processing on the host while the ACCL call is being executed in the FPGA.