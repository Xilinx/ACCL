..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

.. _simulation_section:

##################################
ACCL Emulation Flow for PL Kernels 
##################################

The ACCL driver provides a single entry point to both ACCL hardware and simulator/emulator from the host but 
running a kernel-driven application against the emulator or simulator is slightly more involved. This is for 
two reasons:

* PL kernels themselves can't perform the CCLO initialization therefore an ACCL driver instance must be created but
* the PL kernels don't call the CCLO through the ACCL driver but through the HLS bindings, which transform a call into a sequence of stream reads and writes

The difficulty therefore is connecting the PL kernel streams to the ACCL simulator or emulator simultanously to 
a connection from the ACCL driver to the same. This is achieved through the CCLO Bus Functional Model (BFM)
The CCLO BFM is a software component that has the same stream-like interface as the CCLO, i.e. control and data streams,
such that it can connect to a user PL kernel. Simultaneously, the BFM connects to the CCLO simulator through the same 
mechanism as the ACCL driver. The simulator arbitrates between calls from the driver and BFM just like the client arbiter would,
such that both the driver and PL kernel can issue calls to the CCLO at the same time.

Using the BFM for application testing is illustrated in the example below, which is a simplified version of the vadd test 
in the ACCL repository:

.. code-block:: c++

   //initialize the ACCL driver to perform CCLO configuration
   accl = std::make_unique<ACCL::ACCL>(
      ranks, rank, start_port,
      use_udp ? networkProtocol::UDP : networkProtocol::TCP, 
      rxbuf_count, rxbuf_size
   );

   //allocate float arrays for the HLS function to use
   float src[options.count], dst[options.count];

   //initialize a CCLO BFM and streams as needed
   hlslib::Stream<command_word> callreq, callack;
   hlslib::Stream<stream_word> data_cclo2krnl, data_krnl2cclo;
   std::vector<unsigned int> dest = {9};
   CCLO_BFM cclo(options.start_port, rank, size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
   cclo.run();

   //barrier to make sure all ranks have finished setting up
   MPI_Barrier(MPI_COMM_WORLD);

   //run the hls function, using the global communicator
   vadd_put(   src, dst, options.count, 
               (rank+1)%size,
               accl->get_communicator_addr(), 
               accl->get_arithmetic_config_addr({dataType::float32, dataType::float32}), 
               callreq, callack, 
               data_krnl2cclo, data_cclo2krnl);

   //stop the BFM
   cclo.stop();

Here we create an ACCL driver instance, we create necessary control and data streams using the `hlslib <https://github.com/definelicht/hlslib>`_ HLS library, 
instantiate a BFM connecting to the streams, then execute the user's kernel function against the streams.
Internally the function utilizes the ACCL HLS bindings to call the CCLO and exchange data with it through the streams.
Once the function has executed, we stop the BFM.