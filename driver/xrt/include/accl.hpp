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

#pragma once

#include "accl/acclrequest.hpp"
#include "accl/arithconfig.hpp"
#include "accl/buffer.hpp"
#include "accl/cclo.hpp"
#include "accl/communicator.hpp"
#include "accl/constants.hpp"
#include "accl/xrtbuffer.hpp"
#include "accl/xrtbufferp2p.hpp"
#include "accl/xrtdevice.hpp"
#include "accl/simbuffer.hpp"
#include "accl/simdevice.hpp"
#include "accl/coyotebuffer.hpp"
#include "accl/coyotedevice.hpp"
#include <stdexcept>
#include <string>
#include <vector>

/** @file accl.hpp */

namespace ACCL {

/**
 * Main ACCL class that talks to the CCLO on hardware or emulation/simulation.
 *
 */
class ACCL {
public:
  /**
   * Construct a new ACCL object that talks to hardware.
   *
   * @param device        FPGA device on which the CCLO lives
   * @param cclo_ip       The CCLO kernel on the FPGA
   * @param hostctrl_ip   The hostctrl kernel on the FPGA
   * @param devicemem     Memory bank of device memory
   * @param rxbufmem      Memory banks of rxbuf memory
   * @param arith_config  Arithmetic configuration to use
   */
  ACCL(xrt::device &device, xrt::ip &cclo_ip, xrt::kernel &hostctrl_ip, 
       int devicemem, const std::vector<int> &rxbufmem, 
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);

  /**
   * Construct a new ACCL object that talks to the ACCL emulator/simulator.
   *
   * @param start_port    First port to use to connect to the ACCL emulator/
   *                      simulator
   * @param local_rank    Rank of this process
   * @param arith_config  Arithmetic configuration to use
   */
  ACCL(unsigned int start_port, unsigned int local_rank, 
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);

  /**
   * Construct a new ACCL object on Coyote.
   *
   * @param dev           Coyote device object
   * @param arith_config  Arithmetic configuration to use
   */
  ACCL(CoyoteDevice *dev,
       const arithConfigMap &arith_config = DEFAULT_ARITH_CONFIG);
  
  /**
   * Destroy the ACCL object. Automatically deinitializes the CCLO.
   *
   */
  ~ACCL();

  /**
   * Performs a soft reset of the CCLO.
  */
  void soft_reset();

  /**
   * Deinitializes the CCLO.
   *
   */
  void deinit();


  /**
   * Initializes ACCL 
  */
  void initialize(const std::vector<rank_t> &ranks, int local_rank,
                  int n_egr_rx_bufs = 16, addr_t egr_rx_buf_size = 1024, 
                  addr_t max_egr_size = 1024, addr_t max_rndzv_size = 32*1024);

  /**
   * Get the return code of the last ACCL call.
   *
   * @return val_t The return code
   */
  [[ deprecated ]]
  val_t get_retcode() { return this->cclo->read(CCLO_ADDR::RETCODE_OFFSET); }

  /**
   * Get the hardware id from the FPGA.
   *
   * @return val_t The hardware id
   */
  val_t get_hwid() { return this->cclo->read(CCLO_ADDR::IDCODE_OFFSET); }

  /**
   * Parse the hardware id from the FPGA.
   *
   */
  void parse_hwid();

  /**
   * Set the timeout of ACCL calls.
   *
   * @param value          Timeout in miliseconds
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it will
   *                       start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *set_timeout(unsigned int value, bool run_async = false,
                           std::vector<ACCLRequest *> waitfor = {});

  /**
   * Set the threshold for eager/rendezvous decision.
   *
   * @param value      Threshold in bytes
   * @param run_async  Run the ACCL call asynchronously.
   * @param waitfor    ACCL call will wait for these operations before it will
   *                   start. Currently not implemented.
   * @return CCLO*     CCLO object that can be waited on and passed to waitfor;
   *                   nullptr if run_async is false.
   */
  ACCLRequest *set_rendezvous_threshold(unsigned int value, bool run_async = false,
                    std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the nop operation on the FPGA.
   *
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it will
   *                       start.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *nop(bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the send operation on the FPGA.
   *
   * @param srcbuf         Buffer that contains the data to be send. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements in buffer to send.
   * @param dst            Destination rank to send data to.
   * @param tag            Tag of send operation.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param stream_flags   Stream flags to use.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *send(BaseBuffer &srcbuf, unsigned int count, unsigned int dst,
                    unsigned int tag = TAG_ANY, communicatorId comm_id = GLOBAL_COMM,
                    bool from_fpga = false,
                    dataType compress_dtype = dataType::none, bool run_async = false,
                    std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the send operation on the FPGA using the data stream of the CCLO as input.
   *
   * @param src_data_type  Data type of the input.
   * @param count          Amount of elements in buffer to send.
   * @param dst            Destination rank to send data to.
   * @param tag            Tag of send operation.
   * @param comm_id        Index of communicator to use.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *send(dataType src_data_type, unsigned int count, unsigned int dst,
                    unsigned int tag = TAG_ANY, communicatorId comm_id = GLOBAL_COMM,
                    dataType compress_dtype = dataType::none, bool run_async = false,
                    std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs a one-sided put to a stream on a remote FPGA.
   *
   * @param srcbuf         Buffer that contains the data to be send. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements in buffer to send.
   * @param dst            Destination rank to send data to.
   * @param stream_id      ID of target stream on destination rank. IDs 0-8 are reserved, throws exception if set in this range.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param stream_flags   Stream flags to use. Note that only OP0_STREAM is relevant.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *stream_put(BaseBuffer &srcbuf, unsigned int count,
                          unsigned int dst, unsigned int stream_id, communicatorId comm_id = GLOBAL_COMM,
                          bool from_fpga = false, dataType compress_dtype = dataType::none,
                          bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

    /**
   * Performs a one-sided put to a stream on a remote FPGA using the data stream of the CCLO as input.
   *
   * @param src_data_type  Data type of the input.
   * @param count          Amount of elements in buffer to send.
   * @param dst            Destination rank to send data to.
   * @param stream_id      ID of target stream on destination rank. IDs 0-8 are reserved, throws exception if set in this range.
   * @param comm_id        Index of communicator to use.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *stream_put(dataType src_data_type, unsigned int count,
                          unsigned int dst, unsigned int stream_id, communicatorId comm_id = GLOBAL_COMM,
                          dataType compress_dtype = dataType::none, bool run_async = false,
                          std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the receive operation on the FPGA.
   *
   * @param dstbuf         Buffer where the data should be stored to. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements to receive.
   * @param src            Source rank to receive data from.
   * @param tag            Tag of receive operation.
   * @param comm_id        Index of communicator to use.
   * @param to_fpga        Set to true if the data will be used on the FPGA
   *                       only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *recv(BaseBuffer &dstbuf, unsigned int count, unsigned int src,
                    unsigned int tag = TAG_ANY, communicatorId comm_id = GLOBAL_COMM,
                    bool to_fpga = false,
                    dataType compress_dtype = dataType::none, bool run_async = false,
                    std::vector<ACCLRequest *> waitfor = {});

    /**
   * Performs the receive operation on the FPGA and directs the received data to
   * the data stream of the CCLO.
   *
   * @param dst_data_type  Data Type of the received data.
   * @param count          Amount of elements to receive.
   * @param src            Source rank to receive data from.
   * @param tag            Tag of receive operation.
   * @param comm_id        Index of communicator to use.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *recv(dataType dst_data_type, unsigned int count, unsigned int src,
                    unsigned int tag = TAG_ANY, communicatorId comm_id = GLOBAL_COMM,
                    dataType compress_dtype = dataType::none, bool run_async = false,
                    std::vector<ACCLRequest *> waitfor = {});

  /**
   * Copy a buffer on the FPGA.
   *
   * @param srcbuf         Buffer that contains the data to be copied. Create a
   *                       buffer using ACCL::create_buffer.
   * @param dstbuf         Buffer where the data should be stored to. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements in buffer to copy.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the copied data will be used on the
   *                       FPGA only.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *copy(BaseBuffer &srcbuf, BaseBuffer &dstbuf, unsigned int count,
                    bool from_fpga = false, bool to_fpga = false,
                    bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Copy a buffer on the FPGA.
   *
   * @param dstbuf         Buffer where the data should be stored to. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements in buffer to copy.
   * @param to_fpga        Set to true if the data is already on the FPGA.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *copy_from_stream(BaseBuffer &dstbuf, unsigned int count,
                    bool to_fpga = false,
                    bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Copy a buffer on the FPGA.
   *
   * @param srcbuf         Buffer that contains the data to be copied. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements in buffer to copy.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *copy_to_stream(BaseBuffer &srcbuf, unsigned int count,
                    bool from_fpga = false,
                    bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Copy a buffer on the FPGA.
   *
   * @param dst_data_type  Data type of input and output to stream.
   * @param count          Amount of elements in buffer to copy.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *copy_from_to_stream(dataType dst_data_type, unsigned int count,
                    bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Perform reduce operation on two buffers on the FPGA.
   *
   * @param count           Amount of elements to perform reduce operation on.
   * @param function        Reduce operation to perform.
   * @param val1            First buffer that should be used for reduce
   *                        operation. Create a buffer using
   *                        ACCL::create_buffer.
   * @param val2            Second buffer that should be used for reduce
   *                        operation. Create a buffer using
   *                        ACCL::create_buffer.
   * @param result          Buffer where the result should be stored to. Create
   *                        a buffer using ACCL::create_buffer.
   * @param val1_from_fpga  Set to true if the data of the first buffer is
   *                        already on the FPGA.
   * @param val2_from_fpga  Set to true if the data of the second buffer is
   *                        already on the FPGA.
   * @param to_fpga         Set to true if the copied data will be used on the
   *                        FPGA only.
   * @param run_async       Run the ACCL call asynchronously.
   * @param waitfor         ACCL call will wait for these operations before it
   *                        will start. Currently not implemented.
   * @return ACCLRequest*   Request object used for waiting and checking for operation
   *                        status
   */
  ACCLRequest *combine(unsigned int count, reduceFunction function, BaseBuffer &val1,
                      BaseBuffer &val2, BaseBuffer &result,
                      bool val1_from_fpga = false, bool val2_from_fpga = false,
                      bool to_fpga = false, bool run_async = false,
                      std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the broadcast operation on the FPGA.
   *
   * @param buf            Buffer that should contain the same data as the root
   *                       after the operation. Create a buffer using
   *                       ACCL::create_buffer.
   * @param count          Amount of elements in buffer to broadcast.
   * @param root           Rank to broadcast the data from.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the copied data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *bcast(BaseBuffer &buf, unsigned int count, unsigned int root,
                     communicatorId comm_id = GLOBAL_COMM, bool from_fpga = false,
                     bool to_fpga = false, dataType compress_dtype = dataType::none,
                     bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the scatter operation on the FPGA.
   *
   * @param sendbuf        Buffer of count × world size elements that contains
   *                       the data to be scattered. Create a buffer using
   *                       ACCL::create_buffer. You can pass a DummyBuffer on
   *                       non-root ranks.
   * @param recvbuf        Buffer of count elements where the scattered data
   *                       should be stored. Create a buffer using
   *                       ACCL::create_buffer.
   * @param count          Amount of elements to scatter per rank.
   * @param root           Rank to scatter the data from.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the scattered data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *scatter(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                       unsigned int root, communicatorId comm_id = GLOBAL_COMM,
                       bool from_fpga = false, bool to_fpga = false,
                       dataType compress_dtype = dataType::none,
                       bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the gather operation on the FPGA.
   *
   * @param sendbuf        Buffer of count elements that contains the data to
   *                       be gathered. Create a buffer using
   *                       ACCL::create_buffer.
   * @param recvbuf        Buffer of count × world size elements to where the
   *                       data should be gathered. Create a buffer using
   *                       ACCL::create_buffer. You can pass a DummyBuffer on
   *                       non-root ranks.
   * @param count          Amount of elements to gather per rank.
   * @param root           Rank to gather the data to.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the gathered data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *gather(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                      unsigned int root, communicatorId comm_id = GLOBAL_COMM,
                      bool from_fpga = false, bool to_fpga = false,
                      dataType compress_dtype = dataType::none, bool run_async = false,
                      std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the allgather operation on the FPGA.
   *
   * @param sendbuf        Buffer of count elements that contains the data to
   *                       be gathered. Create a buffer using
   *                       ACCL::create_buffer.
   * @param recvbuf        Buffer of count × world size elements to where the
   *                       data should be gathered. Create a buffer using
   *                       ACCL::create_buffer.
   * @param count          Amount of elements to gather per rank.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the gathered data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *allgather(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                         communicatorId comm_id = GLOBAL_COMM, bool from_fpga = false,
                         bool to_fpga = false,
                         dataType compress_dtype = dataType::none,
                         bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the reduce operation on the FPGA.
   *
   * @param sendbuf        Buffer that contains the data to be reduced. Create a
   *                       buffer using ACCL::create_buffer.
   * @param recvbuf        Buffer to where the data should be reduced. Create a
   *                       buffer using ACCL::create_buffer. You can pass a
   *                       DummyBuffer on non-root ranks.
   * @param count          Amount of elements to reduce.
   * @param root           Rank to reduce the data to.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *reduce(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                      unsigned int root, reduceFunction func,
                      communicatorId comm_id = GLOBAL_COMM, bool from_fpga = false,
                      bool to_fpga = false, dataType compress_dtype = dataType::none,
                      bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the reduce operation on the FPGA from data in local stream
   *
   * @param src_data_type  Data type of the input.
   * @param recvbuf        Buffer to where the data should be reduced. Create a
   *                       buffer using ACCL::create_buffer. You can pass a
   *                       DummyBuffer on non-root ranks.
   * @param count          Amount of elements to reduce.
   * @param root           Rank to reduce the data to.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *reduce(dataType src_data_type, BaseBuffer &recvbuf, unsigned int count, 
                      unsigned int root, reduceFunction func, 
                      communicatorId comm_id = GLOBAL_COMM,
                      bool to_fpga = false, dataType compress_dtype = dataType::none,
                      bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the memory to stream reduce operation on the FPGA
   *
   * @param sendbuf        Buffer that contains the data to be reduced. Create a
   *                       buffer using ACCL::create_buffer.
   * @param dst_data_type  Data type of the output.
   * @param count          Amount of elements to reduce.
   * @param root           Rank to reduce the data to.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *reduce(BaseBuffer &sendbuf, dataType dst_data_type, unsigned int count,
                      unsigned int root, reduceFunction func,
                      communicatorId comm_id = GLOBAL_COMM, bool from_fpga = false,
                      dataType compress_dtype = dataType::none,
                      bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the stream to stream reduce operation on the FPGA 
   *
   * @param src_data_type  Data type of the input.
   * @param dst_data_type  Data type of the output.
   * @param count          Amount of elements to reduce.
   * @param root           Rank to reduce the data to.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *reduce(dataType src_data_type, dataType dst_data_type, unsigned int count, 
                      unsigned int root, reduceFunction func, 
                      communicatorId comm_id = GLOBAL_COMM,
                      dataType compress_dtype = dataType::none,
                      bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the allreduce operation on the FPGA.
   *
   * @param sendbuf        Buffer that contains the data to be reduced. Create a
   *                       buffer using ACCL::create_buffer.
   * @param recvbuf        Buffer to where the data should be reduced. Create a
   *                       buffer using ACCL::create_buffer.
   * @param count          Amount of elements to reduce.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *allreduce(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                         reduceFunction func, communicatorId comm_id = GLOBAL_COMM,
                         bool from_fpga = false, bool to_fpga = false,
                         dataType compress_dtype = dataType::none,
                         bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs the reduce_scatter operation on the FPGA.
   *
   * @param sendbuf        Buffer of count × world size elements that contains
   *                       the data to be reduced. Create a buffer using
   *                       ACCL::create_buffer.
   * @param recvbuf        Buffer of count elements to where the data should be
   *                       reduced. Create a buffer using ACCL::create_buffer.
   * @param count          Amount of elements to reduce per rank.
   * @param func           Reduce function to use.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the reduced data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *reduce_scatter(BaseBuffer &sendbuf, BaseBuffer &recvbuf,
                              unsigned int count, reduceFunction func,
                              communicatorId comm_id = GLOBAL_COMM,
                              bool from_fpga = false, bool to_fpga = false,
                              dataType compress_dtype = dataType::none,
                              bool run_async = false,
                              std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs an alltoall shuffle operation on the FPGA.
   *
   * @param sendbuf        Buffer of count elements that contains the data to
   *                       be shuffled. Create a buffer using ACCL::create_buffer.
   * @param recvbuf        Buffer of count × world size elements to where the
   *                       data should be gathered. Create a buffer using
   *                       ACCL::create_buffer.
   * @param count          Amount of elements to shuffle per rank.
   * @param comm_id        Index of communicator to use.
   * @param from_fpga      Set to true if the data is already on the FPGA.
   * @param to_fpga        Set to true if the gathered data will be used on the
   *                       FPGA only.
   * @param compress_dtype Datatype to compress buffers to over ethernet.
   * @param run_async      Run the ACCL call asynchronously.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   */
  ACCLRequest *alltoall(BaseBuffer &sendbuf, BaseBuffer &recvbuf, unsigned int count,
                         communicatorId comm_id = GLOBAL_COMM, bool from_fpga = false,
                         bool to_fpga = false,
                         dataType compress_dtype = dataType::none,
                         bool run_async = false, std::vector<ACCLRequest *> waitfor = {});

  /**
   * Performs a barrier on the FPGA.
   *
   * @param comm_id        Index of communicator to use.
   * @param waitfor        ACCL call will wait for these operations before it
   *                       will start. Currently not implemented.
   * @return ACCLRequest*  Request object used for waiting and checking for operation
   *                       status
   *
   */
ACCLRequest *barrier(communicatorId comm_id = GLOBAL_COMM,
               std::vector<ACCLRequest *> waitfor = {});

  /**
   * Wait for an ACCLRequest. Waits (blocking) the caller while the
   * request completes
   * 
   * @param request Reference to the ACCLRequest to wait for
   */
  void wait(ACCLRequest *request);

  /**
   * Wait for an ACCLRequest. Waits (blocking) the caller while the
   * request completes, or until time out
   * 
   * @param request Reference to the ACCLRequest to wait for
   * @param timeout Time in milli seconds to wait for
   * @return true   If the request complete
   * @return false  If the request did not complete
   */
  bool wait(ACCLRequest *request, std::chrono::milliseconds timeout);

  /**
   * Check if given request is completed
   * 
   * @param request Reference to the ACCLRequest to test
   * @return true   If the request complete
   * @return false  If the request did not complete
   */
  bool test(ACCLRequest *request);

  /**
   * Get duration of call associated to request
   * 
   * @param request Reference to the ACCLRequest to test
   * @return uint64_t  Call duration in nanoseconds
   */
  uint64_t get_duration(ACCLRequest *request);

  /**
   * Check if given request is completed
   * 
   * @param request Reference to the ACCLRequest to free
   */
  void free_request(ACCLRequest *request);

  /**
   * Check if ACCL is being run in simulated mode or not.
   *
   * @return true  ACCL is running on an emulator or simulator.
   * @return false ACCL is running on hardware.
   */
  bool is_simulated() const { return sim_mode; }

  std::vector<rank_t> get_comm_group(communicatorId comm_id);

  unsigned int get_comm_rank(communicatorId comm_id);

  communicatorId create_communicator(const std::vector<rank_t> &ranks,
                                     int local_rank);

  /**
   * Construct a new buffer object without an existing host buffer.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(size_t length, dataType type) {
    return create_buffer<dtype>(length, type, _devicemem);
  }

  /**
   * Construct a new buffer object without an existing host buffer on the
   * specified memory bank.
   *
   * Only use this function if you want to store the buffer on a different
   * memory bank than the devicemem bank specified during construction.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @param mem_grp             Memory bank to allocate buffer on.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(size_t length, dataType type, unsigned mem_grp) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(new SimBuffer<dtype>(
          length, type, static_cast<SimDevice *>(cclo)->get_context()));
    } else if (cclo->get_device_type() == CCLO::xrt_device) {
      return std::unique_ptr<Buffer<dtype>>(new XRTBuffer<dtype>(
          length, type, *(static_cast<XRTDevice *>(cclo)->get_device()), (xrt::memory_group)mem_grp));
    } else {
      return std::unique_ptr<Buffer<dtype>>(new CoyoteBuffer<dtype>(
          length, type, cclo));
    }
  }

  /**
   * Construct a new buffer object from an existing host pointer.
   *
   * On hardware it is required that the host pointer is aligned to 4096 bytes.
   * If a non-aligned host pointer is provided and ACCL is running on hardware,
   * ACCL will keep it's own aligned host buffer, and copy between the
   * unaligned and aligned host buffers when required. It is recommended to
   * provide an aligned host pointer to avoid unnecessary memory copies.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param host_buffer         The host pointer containing the data.
   * @param length              Amount of elements in the host buffer.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(dtype *host_buffer,
                                               size_t length, dataType type) {
    return create_buffer(host_buffer, length, type, _devicemem);
  }

  /**
   * Construct a new buffer object from an existing host pointer on the
   * specified memory bank.
   *
   * Only use this function if you want to store the buffer on a different
   * memory bank than the devicemem bank specified during construction.
   *
   * On hardware it is required that the host pointer is aligned to 4096 bytes.
   * If a non-aligned host pointer is provided and ACCL is running on hardware,
   * ACCL will keep it's own aligned host buffer, and copy between the
   * unaligned and aligned host buffers when required. It is recommended to
   * provide an aligned host pointer to avoid unnecessary memory copies.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param host_buffer         The host pointer containing the data.
   * @param length              Amount of elements in the host buffer.
   * @param type                ACCL datatype of the buffer.
   * @param mem_grp             Memory bank to allocate buffer on.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(dtype *host_buffer,
                                               size_t length, dataType type,
                                               unsigned mem_grp) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(
          new SimBuffer<dtype>(host_buffer, length, type,
                               static_cast<SimDevice *>(cclo)->get_context()));
    } else if(cclo->get_device_type() == CCLO::xrt_device ){
      return std::unique_ptr<Buffer<dtype>>(new XRTBuffer<dtype>(
          host_buffer, length, type, *(static_cast<XRTDevice *>(cclo)->get_device()), (xrt::memory_group)mem_grp));
    }
    return std::unique_ptr<Buffer<dtype>>(nullptr);
  }

  /**
   * Construct a new buffer object from an existing BO buffer.
   *
   * When using an ACCL emulator or simulator, this function can be used to
   * pass a simulated BO buffer from the Vitis emulator and use the Vitis
   * emulator together with the ACCL emulator. In this case, ACCL will also
   * create a new internal simulated BO buffer to copy data between the
   * simulated BO buffer and the simulated ACCL buffer when required.
   *
   * When running on hardware, ACCL will simply use this BO buffer internally,
   * instead of allocating a new one.
   *
   * @tparam dtype              Datatype of the buffer.
   * @param bo                  The BO buffer to use.
   * @param length              Amount of elements in the BO buffer.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer(xrt::bo &bo, size_t length,
                                               dataType type) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(
          new SimBuffer<dtype>(bo, *(static_cast<SimDevice *>(cclo)->get_device()), length, type,
                               static_cast<SimDevice *>(cclo)->get_context()));
    } else {
      return std::unique_ptr<Buffer<dtype>>(
          new XRTBuffer<dtype>(bo, length, type));
    }
  }

  /**
   * Construct a new p2p buffer object.
   *
   * Will create a normal buffer when running in simulated mode.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer_p2p(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated P2P buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer_p2p(size_t length,
                                                   dataType type) {
    return create_buffer_p2p<dtype>(length, type, _devicemem);
  }

  /**
   * Construct a new p2p buffer object on the specified memory bank.
   *
   * Will create a normal buffer when running in simulated mode.
   *
   * Only use this function if you want to store the buffer on a different
   * memory bank than the devicemem bank specified during construction.
   *
   * Note that when running in simulated mode, this constructor will not create
   * an underlying simulated BO buffer. If you need this functionality, use
   * create_buffer_p2p(xrt::bo &, size_t, dataType).
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @param mem_grp             Memory bank to allocate buffer on.
   * @return std::unique_ptr<Buffer<dtype>> The allocated P2P buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer_p2p(size_t length, dataType type,
                                                   unsigned mem_grp) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(new SimBuffer<dtype>(
          length, type, static_cast<SimDevice *>(cclo)->get_context()));
    } else if(cclo->get_device_type() == CCLO::xrt_device ){
      return std::unique_ptr<Buffer<dtype>>(new XRTBufferP2P<dtype>(
          length, type, *(static_cast<XRTDevice *>(cclo)->get_device()), (xrt::memory_group)mem_grp));
    } else {
      //for Coyote there's no concept of a p2p buffer
      throw std::runtime_error("p2p buffers not supported in Coyote");
    }
  }

  /**
   * Construct a new p2p buffer object from an existing P2P BO buffer.
   *
   * If you do not pass a non-P2P BO buffer, data will not be copied correctly
   * from and to the FPGA.
   *
   * Will create a normal buffer when running in simulated mode. See the notes
   * of create_buffer(xrt::bo &, size_t, dataType) about using BO buffers in
   * simulated mode.
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated P2P buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_buffer_p2p(xrt::bo &bo, size_t length,
                                                   dataType type) {
    if (sim_mode) {
      return std::unique_ptr<Buffer<dtype>>(
          new SimBuffer<dtype>(bo, *(static_cast<SimDevice *>(cclo)->get_device()), length, type,
                               static_cast<SimDevice *>(cclo)->get_context()));
    } else {
      return std::unique_ptr<Buffer<dtype>>(
          new XRTBufferP2P<dtype>(bo, length, type));
    }
  }

  /**
   * Construct a new coyote buffer object without an existing host buffer 
   *
   * Coyote buffer object doesn't have a notion of memory banks
   *
   *
   * @tparam dtype              Datatype of the buffer.
   * @param length              Amount of elements to allocate for.
   * @param type                ACCL datatype of the buffer.
   * @return std::unique_ptr<Buffer<dtype>> The allocated buffer.
   */
  template <typename dtype>
  std::unique_ptr<Buffer<dtype>> create_coyotebuffer(size_t length, dataType type) {
    if (sim_mode) {
      throw std::runtime_error("create_coyotebuffer sim_mode unsupported!!!");
    } else {
      return std::unique_ptr<Buffer<dtype>>(new CoyoteBuffer<dtype>(length, type, static_cast<CoyoteDevice *>(cclo)));
    }
  }

  /**
   * Dump the content of the exchange memory to a string.
   *
   * @return std::string Content of the exchange memory.
   */
  std::string dump_exchange_memory();

  /**
   * Dump the content of the Eager-mode RX buffers to a string for the first n_egr_rx_bufs buffers.
   *
   * @param n_egr_rx_bufs        Amount of buffers to dump the content of.
   * @param dump_data    Dump buffer contents along with metadata.
   * @return std::string Content of the Eager-mode RX buffers.
   */
  std::string dump_eager_rx_buffers(size_t n_egr_rx_bufs, bool dump_data=true);

  /**
   * Dump the content of all Eager-mode RX buffers to a string.
   *
   * @return std::string Content of all Eager-mode RX buffers.
   */
  std::string dump_eager_rx_buffers(bool dump_data=true) {
    if (cclo->read(CCLO_ADDR::NUM_EGR_RX_BUFS_OFFSET) != eager_rx_buffers.size()) {
      throw std::runtime_error("CCLO inconsistent");
    }
    return dump_eager_rx_buffers(eager_rx_buffers.size(), dump_data);
  }

  /**
   * Dump the content of the communicator to a string.
   *
   * @return std::string Content of the communicator.
   */
  std::string dump_communicator();

  /**
   * Return CCLO address of communicator.
   *
   * @param ACCL::communicatorId Numerical ID of the target communicator.
   * @return addr_t Address of the communicator in CCLO memory.
   */
  addr_t get_communicator_addr(communicatorId comm_id=GLOBAL_COMM);

  /**
   * Return CCLO address of arithmetic config.
   *
   * @param unsigned int Numerical ID of the target arithmetic configuration.
   * @return addr_t Address of the arithmetic configuration in CCLO memory.
   */
  addr_t get_arithmetic_config_addr(std::pair<dataType, dataType> id);

  /**
   * Retrieve the devicemem memory bank.
   *
   * @return int The devicemem memory bank
   */
  int devicemem() { return _devicemem; }

  /**
   * Open port on network interface of FPGA to exchange ACCL message.
   *
   * @param comm_id Numerical ID of the target communicator.
   */
  void open_port(communicatorId comm_id = GLOBAL_COMM);

  /**
   * Open connections with other ranks.
   *
   * @param comm_id Numerical ID of the target communicator.
   */
  void open_con(communicatorId comm_id = GLOBAL_COMM);

  /**
   * Close connections with other ranks.
   *
   * @param comm_id Numerical ID of the target communicator.
   */
  void close_con(communicatorId comm_id = GLOBAL_COMM);

private:
  CCLO *cclo{};
  // Supported types and corresponding arithmetic config
  arithConfigMap arith_config;
  // Address to put new configurations like arithmetic configs
  // and communicators
  addr_t current_config_address{};
  // RX spare buffers for eager mode
  addr_t max_eager_msg_size{};
  std::vector<Buffer<int8_t> *> eager_rx_buffers;
  addr_t eager_rx_buffer_size{};
  // Spare buffers for use in rendezvous reduces
  addr_t max_rndzv_msg_size{};
  std::vector<Buffer<int8_t> *> utility_spares;
  // List of communicators, to which users will add
  std::vector<Communicator> communicators;
  // safety checks
  bool check_return_value_flag{};
  bool ignore_safety_checks{};
  // TODO: use description to gather info about where to allocate spare buffers
  // flag to indicate whether we've finished config
  bool config_rdy{};
  // flag to indicate whether we're simulating
  const bool sim_mode;
  // memory banks for hardware
  const int _devicemem;
  const std::vector<int> rxbufmem;

  ACCLRequest *copy(BaseBuffer *srcbuf, BaseBuffer *dstbuf, unsigned int count,
                 bool from_fpga, bool to_fpga, streamFlags stream_flags,
                 dataType data_type, bool run_async,
                 std::vector<ACCLRequest *> waitfor);

  void configure_arithmetic();

  void setup_eager_rx_buffers(size_t n_egr_rx_bufs, addr_t egr_rx_buf_size,
                        const std::vector<int> &devicemem);
  void setup_eager_rx_buffers(size_t n_egr_rx_bufs, addr_t egr_rx_buf_size, int devicemem) {
    std::vector<int> mems = {devicemem};
    return setup_eager_rx_buffers(n_egr_rx_bufs, egr_rx_buf_size, mems);
  }

  void setup_rendezvous_spare_buffers(addr_t rndzv_spare_buf_size, const std::vector<int> &devicemem);
  void setup_rendezvous_spare_buffers(addr_t rndzv_spare_buf_size, int devicemem) {
    std::vector<int> mems = {devicemem};
    return setup_rendezvous_spare_buffers(rndzv_spare_buf_size, mems);
  }

  void configure_tuning_parameters();

  void check_return_value(const std::string function_name, ACCLRequest *handle);

  void prepare_call(CCLO::Options &options);

  ACCLRequest *call_async(CCLO::Options &options);

  ACCLRequest *call_sync(CCLO::Options &options);

  void set_max_eager_msg_size(unsigned int value);

  void set_max_rendezvous_msg_size(unsigned int value);

  void configure_communicator(const std::vector<rank_t> &ranks, int local_rank);
};

} // namespace ACCL
