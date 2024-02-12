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

#include <utility.hpp>
#include <fixture.hpp>
#include <tclap/CmdLine.h>

#define FLOAT32RTOL 0.001
#define FLOAT32ATOL 0.005
// Set the tolerance for compressed datatypes high enough, since we do currently
// not replicate the float32 -> float16 conversion for our reference results
#define FLOAT16RTOL 0.005
#define FLOAT16ATOL 0.05

TEST_F(ACCLTest, test_copy){
  if(::size > 1){
    GTEST_SKIP() << "Skipping single-node test on multi-node setup";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  accl->copy(*op_buf, *res_buf, count);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*op_buf)[i], (*res_buf)[i]);
  }
}

TEST_F(ACCLTest, test_copy_stream) {
  if(::size > 1){
    GTEST_SKIP() << "Skipping single-node test on multi-node setup";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  accl->copy_to_stream(*op_buf, count, false);
  accl->copy_from_stream(*res_buf, count, false);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*op_buf)[i], (*res_buf)[i]);
  }
}

TEST_F(ACCLTest, test_copy_p2p) {
  if(::size > 1){
    GTEST_SKIP() << "Skipping single-node test on multi-node setup";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> p2p_buf;
  try {
    p2p_buf = accl->create_buffer_p2p<float>(count, dataType::float32);
  } catch (const std::bad_alloc &e) {
    std::cout << "Can't allocate p2p buffer (" << e.what() << "). "
              << "This probably means p2p is disabled on the FPGA.\n"
              << "Skipping p2p test..." << std::endl;
    return;
  }
  random_array(op_buf->buffer(), count);

  accl->copy(*op_buf, *p2p_buf, count);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*op_buf)[i], (*p2p_buf)[i]);
  }
}

TEST_P(ACCLFuncTest, test_combine) {
  if(::size > 1){
    GTEST_SKIP() << "Skipping single-node test on multi-node setup";
  }
  if((GetParam() != reduceFunction::SUM) && (GetParam() != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  auto op_buf1 = accl->create_buffer<float>(count, dataType::float32);
  auto op_buf2 = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf1->buffer(), count);
  random_array(op_buf2->buffer(), count);

  accl->combine(count, GetParam(), *op_buf1, *op_buf2, *res_buf);

  float ref, res;
  for (unsigned int i = 0; i < count; ++i) {
    if(GetParam() == reduceFunction::SUM){
      ref = (*op_buf1)[i] + (*op_buf2)[i];
      res = (*res_buf)[i];
    } else if(GetParam() == reduceFunction::MAX){
      ref = ((*op_buf1)[i] > (*op_buf2)[i]) ? (*op_buf1)[i] : (*op_buf2)[i];
      res = (*res_buf)[i];
    }
    EXPECT_FLOAT_EQ(ref, res);
  }
}

TEST_F(ACCLTest, test_sendrcv_basic) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = ::rank + 1;
  int prev_rank = ::rank - 1;

  if(::rank % 2 == 0){
    if(next_rank < ::size){
      test_debug("Sending data on " + std::to_string(::rank) + " to " +
                    std::to_string(next_rank) + "...", options);
      accl->send(*op_buf, count, next_rank, 0);
    }
  } else {
      test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                    std::to_string(prev_rank) + "...", options);
      accl->recv(*res_buf, count, prev_rank, 0);
  }

  if(::rank % 2 == 1){
    test_debug("Sending data on " + std::to_string(::rank) + " to " +
                  std::to_string(prev_rank) + "...", options);
    accl->send(*res_buf, count, prev_rank, 1);
  } else {
    if(next_rank < ::size){
      test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                    std::to_string(next_rank) + "...", options);
      accl->recv(*res_buf, count, next_rank, 1);
    }
  }

  if(next_rank < ::size){
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
    }
  } else {
    SUCCEED();
  }
}

TEST_F(ACCLTest, test_sendrcv_bo) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  if(!options.test_xrt_simulator) {
    GTEST_SKIP() << "Skipping xrt::bo test. We are not running on hardware and "
                 "XCL emulation is disabled. Make sure XILINX_VITIS and "
                 "XCL_EMULATION_MODE are set.";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  // Initialize bo
  float *data =
      static_cast<float *>(std::aligned_alloc(4096, count *sizeof(float)));
  float *validation_data =
      static_cast<float *>(std::aligned_alloc(4096, count *sizeof(float)));
  random_array(data, count);

  xrt::bo send_bo(dev, data, count *sizeof(float), accl->devicemem());
  xrt::bo recv_bo(dev, validation_data, count *sizeof(float),
                  accl->devicemem());
  auto op_buf = accl->create_buffer<float>(send_bo, count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(recv_bo, count, dataType::float32);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Syncing buffers...", options);
  send_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*op_buf, count, prev_rank, 0, GLOBAL_COMM, true);

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, true);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*op_buf, count, next_rank, 1, GLOBAL_COMM, true);

  accl->copy(*op_buf, *res_buf, count, true, true);

  recv_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(validation_data[i], data[i]);
  }

  std::free(data);
  std::free(validation_data);
}

TEST_F(ACCLTest, test_sendrcv) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*res_buf, count, prev_rank, 0);

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*res_buf, count, prev_rank, 1);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_P(ACCLSegmentationTest, test_sendrcv_segmentation){
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  unsigned int count_per_segment = options.segment_size / (dataTypeSize.at(dataType::float32) / 8);
  unsigned int multiplier = std::get<0>(GetParam());
  int offset = std::get<1>(GetParam());
  unsigned int count;
  if((count_per_segment * multiplier + offset) <= 0) {
    GTEST_SKIP() << "Multiplier/offset resolve non-positive count. ";
  } else {
    count = count_per_segment * multiplier + offset;
  }
  if(((count + count_per_segment -1) / count_per_segment) > options.rxbuf_count){
    GTEST_SKIP() << "Not enough spare buffers for segmentation test. ";
  }
  test_debug("Testing send/recv segmentation with count=" + std::to_string(count), options);
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*res_buf, count, prev_rank, 0);

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*res_buf, count, prev_rank, 1);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
  }
}

TEST_F(ACCLTest, test_sendrcv_stream) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
             std::to_string(next_rank) + "...", options);
  accl->send(*op_buf, count, next_rank, 0);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
             std::to_string(prev_rank) + "...", options);
  accl->recv(dataType::float32, count, prev_rank, 0, GLOBAL_COMM);

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
             std::to_string(prev_rank) + "...", options);
  accl->send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_F(ACCLTest, test_stream_put) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Sending data on " + std::to_string(::rank) + " to stream 0 on " +
             std::to_string(next_rank) + "...", options);
  accl->stream_put(*op_buf, count, next_rank, 9);

  test_debug("Sending data on " + std::to_string(::rank) + " from stream to " +
             std::to_string(prev_rank) + "...", options);
  accl->send(dataType::float32, count, prev_rank, 1, GLOBAL_COMM);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
             std::to_string(next_rank) + "...", options);
  accl->recv(*res_buf, count, next_rank, 1);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
  }

}

TEST_F(ACCLTest, test_sendrcv_compressed) {
  if(::size == 1){
    GTEST_SKIP() << "Skipping send/recv test on single-node setup";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  int next_rank = (::rank + 1) % ::size;
  int prev_rank = (::rank + ::size - 1) % ::size;

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(next_rank) + "...",
             options);
  accl->send(*op_buf, count, next_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->recv(*res_buf, count, prev_rank, 0, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], (*op_buf)[i], FLOAT16RTOL, FLOAT16ATOL));
  }

  test_debug("Sending data on " + std::to_string(::rank) + " to " +
                 std::to_string(prev_rank) + "...",
             options);
  accl->send(*op_buf, count, prev_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  test_debug("Receiving data on " + std::to_string(::rank) + " from " +
                 std::to_string(next_rank) + "...",
             options);
  accl->recv(*res_buf, count, next_rank, 1, GLOBAL_COMM, false,
            dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], (*op_buf)[i], FLOAT16RTOL, FLOAT16ATOL));
  }

}

TEST_P(ACCLRootTest, test_bcast) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  int root = GetParam();
  if (::rank == root) {
    test_debug("Broadcasting data from " + std::to_string(::rank) + "...", options);
    accl->bcast(*op_buf, count, root);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...", options);
    accl->bcast(*res_buf, count, root);
  }

  if (::rank != root) {
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i]);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_bcast_compressed) {
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  int root = GetParam();
  if (::rank == root) {
    test_debug("Broadcasting data from " + std::to_string(::rank) + "...", options);
    accl->bcast(*op_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);
  } else {
    test_debug("Getting broadcast data from " + std::to_string(root) + "...", options);
    accl->bcast(*res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);
  }

  if (::rank != root) {
    for (unsigned int i = 0; i < count; ++i) {
      float res = (*res_buf)[i];
      float ref = (*op_buf)[i];
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_scatter) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count * ::size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * ::size);
  int root = GetParam();
  test_debug("Scatter data from " + std::to_string(::rank) + "...", options);
  accl->scatter(*op_buf, *res_buf, count, root);

  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], (*op_buf)[i +::rank * count]);
  }
}

TEST_P(ACCLRootTest, test_scatter_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count * ::size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count * ::size);
  int root = GetParam();
  test_debug("Scatter data from " + std::to_string(::rank) + "...", options);
  accl->scatter(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);

  for (unsigned int i = 0; i < count; ++i) {
    float res = (*res_buf)[i];
    float ref = (*op_buf)[i +::rank * count];
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_P(ACCLRootTest, test_gather) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  int root = GetParam();
  std::unique_ptr<float> host_op_buf = random_array<float>(count * ::size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count *::rank, count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (::rank == root) {
    res_buf = accl->create_buffer<float>(count *::size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(::rank) + "...", options);
  accl->gather(*op_buf, *res_buf, count, root);

  if (::rank == root) {
    for (unsigned int i = 0; i < count *::size; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootTest, test_gather_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  int root = GetParam();
  std::unique_ptr<float> host_op_buf = random_array<float>(count *::size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count *::rank, count, dataType::float32);
  std::unique_ptr<ACCL::Buffer<float>> res_buf;
  if (::rank == root) {
    res_buf = accl->create_buffer<float>(count *::size, dataType::float32);
  } else {
    res_buf = std::unique_ptr<ACCL::Buffer<float>>(nullptr);
  }

  test_debug("Gather data from " + std::to_string(::rank) + "...", options);
  accl->gather(*op_buf, *res_buf, count, root, GLOBAL_COMM, false, false, dataType::float16);

  if (::rank == root) {
    for (unsigned int i = 0; i < count *::size; ++i) {
      float res = (*res_buf)[i];
      float ref = host_op_buf.get()[i];
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_F(ACCLTest, test_alltoall) {
  if(!options.cyt_rdma){
    GTEST_SKIP() << "Alltoall requires rendezvous support, currently only available on Coyote RDMA backend";
  }
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  std::unique_ptr<float> host_op_buf = random_array<float>(count *::size *::size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count *::size *::rank, count *::size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count *::size, dataType::float32);

  test_debug("Shuffling (all2all) data...", options);
  accl->alltoall(*op_buf, *res_buf, count);

  for (int i = 0; i < ::size; ++i) {
    for (unsigned int j = 0; j < count; ++j) {
      EXPECT_FLOAT_EQ((*res_buf)[i*count+j], host_op_buf.get()[i*::size*count+::rank*count+j]);
    }
  }
}

TEST_F(ACCLTest, test_allgather) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  std::unique_ptr<float> host_op_buf = random_array<float>(count *::size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count *::rank, count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count *::size, dataType::float32);

  test_debug("Gathering data...", options);
  accl->allgather(*op_buf, *res_buf, count);

  for (unsigned int i = 0; i < count *::size; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], host_op_buf.get()[i]);
  }
}

TEST_F(ACCLTest, test_allgather_compressed) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  std::unique_ptr<float> host_op_buf = random_array<float>(count *::size);
  auto op_buf = accl->create_buffer(host_op_buf.get() + count *::rank, count,
                                   dataType::float32);
  auto res_buf = accl->create_buffer<float>(count *::size, dataType::float32);

  test_debug("Gathering data...", options);
  accl->allgather(*op_buf, *res_buf, count, GLOBAL_COMM, false, false,
                 dataType::float16);

  for (unsigned int i = 0; i < count *::size; ++i) {
    EXPECT_TRUE(is_close((*res_buf)[i], host_op_buf.get()[i], FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_F(ACCLTest, test_allgather_comms) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  std::unique_ptr<float> host_op_buf(new float[count *::size]);
  auto op_buf = accl->create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count *::size, dataType::float32);

  for (unsigned int i = 0; i < count *::size; i++) {
    host_op_buf.get()[i] =::rank + i;
  }
  std::fill(res_buf->buffer(), res_buf->buffer() + count *::size, 0);

  test_debug("Setting up communicators...", options);
  auto group = accl->get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl->get_comm_rank(GLOBAL_COMM);
  unsigned int split = group.size() / 2;
  test_debug("Split is " + std::to_string(split), options);
  std::vector<rank_t>::iterator new_group_start;
  std::vector<rank_t>::iterator new_group_end;
  unsigned int new_rank = own_rank;
  bool is_in_lower_part = own_rank < split;
  if (is_in_lower_part) {
    new_group_start = group.begin();
    new_group_end = group.begin() + split;
  } else {
    new_group_start = group.begin() + split;
    new_group_end = group.end();
    new_rank = own_rank - split;
  }
  std::vector<rank_t> new_group(new_group_start, new_group_end);
  communicatorId new_comm = accl->create_communicator(new_group, new_rank);
  test_debug(accl->dump_communicator(), options);
  test_debug("Gathering data... count=" + std::to_string(count) +
                 ", comm=" + std::to_string(new_comm),
             options);
  accl->allgather(*op_buf, *res_buf, count, new_comm);
  test_debug("Validate data...", options);

  unsigned int data_split =
      is_in_lower_part ? count * split : count *::size - count * split;

  for (unsigned int i = 0; i < count *::size; ++i) {
    float res = (*res_buf)[i];
    float ref;
    if (i < data_split) {
      ref = is_in_lower_part ? 0 : split;
      ref += (i / count) + (i % count);
    } else {
      ref = 0.0;
    }
    EXPECT_FLOAT_EQ(res, ref);
  }
}

TEST_F(ACCLTest, test_multicomm) {
  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto group = accl->get_comm_group(GLOBAL_COMM);
  unsigned int own_rank = accl->get_comm_rank(GLOBAL_COMM);
  int errors = 0;
  if (group.size() < 4) {
    GTEST_SKIP() << "Too few ranks for multi-communicator test";
  }
  if (own_rank == 1 || own_rank > 3) {
    EXPECT_TRUE(true);
  }
  std::vector<rank_t> new_group;
  new_group.push_back(group[0]);
  new_group.push_back(group[2]);
  new_group.push_back(group[3]);
  unsigned int new_rank = (own_rank == 0) ? 0 : own_rank - 1;
  communicatorId new_comm = accl->create_communicator(new_group, new_rank);
  //return if we're not involved in the new communicator
  if(own_rank == 1) return;
  //else, start testing
  test_debug(accl->dump_communicator(), options);
  std::unique_ptr<float> host_op_buf = random_array<float>(count);
  auto op_buf = accl->create_buffer(host_op_buf.get(), count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  // start with a send/recv between every pair of ranks in the new communicator
  if (new_rank == 0) {
    accl->send(*op_buf, count, 1, 0, new_comm);
    accl->recv(*res_buf, count, 1, 1, new_comm);
    test_debug("Second recv completed", options);
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else if (new_rank == 1) {
    accl->recv(*res_buf, count, 0, 0, new_comm);
    test_debug("First recv completed", options);
    accl->send(*op_buf, count, 0, 1, new_comm);
  }

  if (new_rank == 1) {
    accl->send(*op_buf, count, 2, 2, new_comm);
    accl->recv(*res_buf, count, 2, 3, new_comm);
    test_debug("Second recv completed", options);
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else if (new_rank == 2) {
    accl->recv(*res_buf, count, 1, 2, new_comm);
    test_debug("First recv completed", options);
    accl->send(*op_buf, count, 1, 3, new_comm);
  }

  if (new_rank == 0) {
    accl->send(*op_buf, count, 2, 4, new_comm);
    accl->recv(*res_buf, count, 2, 5, new_comm);
    test_debug("Second recv completed", options);
    for (unsigned int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ((*res_buf)[i], host_op_buf.get()[i]);
    }
  } else if (new_rank == 2) {
    accl->recv(*res_buf, count, 0, 4, new_comm);
    test_debug("First recv completed", options);
    accl->send(*op_buf, count, 0, 5, new_comm);
  }

  // do an all-reduce on the new communicator
  for (unsigned int i = 0; i < count; ++i) {
    host_op_buf.get()[i] = i;
  }
  accl->allreduce(*op_buf, *res_buf, count, ACCL::reduceFunction::SUM, new_comm);
  for (unsigned int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ((*res_buf)[i], 3*host_op_buf.get()[i]);
  }

  test_debug(accl->dump_communicator(), options);
}

TEST_P(ACCLRootFuncTest, test_reduce) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, *res_buf, count, root, function);

  float res, ref;
  if (::rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
      EXPECT_TRUE(is_close(res, ref, FLOAT32RTOL, FLOAT32ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_compressed) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, *res_buf, count, root, function, GLOBAL_COMM, false,
              false, dataType::float16);

  float res, ref;
  if (::rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
      EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_stream2mem) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Loading stream on::rank" + std::to_string(::rank) + "...", options);
  accl->copy_to_stream(*op_buf, count, false);
  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(dataType::float32, *res_buf, count, root, function);

  float res, ref;
  if (::rank == root) {
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
      EXPECT_FLOAT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_mem2stream) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  random_array(res_buf->buffer(), count);
  res_buf->sync_to_device();

  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(*op_buf, dataType::float32, count, root, function);

  float res, ref;
  if (::rank == root) {
    test_debug("Unloading stream on::rank" + std::to_string(::rank) + "...", options);
    accl->copy_from_stream(*res_buf, count, false);
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
      EXPECT_FLOAT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLRootFuncTest, test_reduce_stream2stream) {
  int root = std::get<0>(GetParam());
  reduceFunction function = std::get<1>(GetParam());
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);
  random_array(res_buf->buffer(), count);
  res_buf->sync_to_device();

  test_debug("Loading stream on::rank" + std::to_string(::rank) + "...", options);
  accl->copy_to_stream(*op_buf, count, false);
  test_debug("Reduce data to " + std::to_string(root) + "...", options);
  accl->reduce(dataType::float32, dataType::float32, count, root, function);

  float res, ref;
  if (::rank == root) {
    test_debug("Unloading stream on::rank" + std::to_string(::rank) + "...", options);
    accl->copy_from_stream(*res_buf, count, false);
    for (unsigned int i = 0; i < count; ++i) {
      res = (*res_buf)[i];
      ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
      EXPECT_FLOAT_EQ(res, ref);
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST_P(ACCLFuncTest, test_reduce_scatter) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count *::size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count *::size);

  test_debug("Reducing data...", options);
  accl->reduce_scatter(*op_buf, *res_buf, count, function);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i+ ::rank *count] : (*op_buf)[i+ ::rank *count] *::size;
    EXPECT_FLOAT_EQ(res, ref);
  }
}

TEST_P(ACCLFuncTest, test_reduce_scatter_compressed) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  unsigned int count_bytes = count * dataTypeSize.at(dataType::float32) / 8;

  auto op_buf = accl->create_buffer<float>(count *::size, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count *::size);

  test_debug("Reducing data...", options);
  accl->reduce_scatter(*op_buf, *res_buf, count, function, GLOBAL_COMM, false, false, dataType::float16);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i+ ::rank *count] : (*op_buf)[i+ ::rank *count] *::size;
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_P(ACCLFuncTest, test_allreduce) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }

  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl->allreduce(*op_buf, *res_buf, count, function);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
    EXPECT_FLOAT_EQ(res, ref);
  }
}

TEST_P(ACCLFuncTest, test_allreduce_compressed) {
  reduceFunction function = GetParam();
  if((function != reduceFunction::SUM) && (function != reduceFunction::MAX)){
    GTEST_SKIP() << "Unrecognized reduction function";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  accl->allreduce(*op_buf, *res_buf, count, function, GLOBAL_COMM, false, false, dataType::float16);

  float res, ref;
  for (unsigned int i = 0; i < count; ++i) {
    res = (*res_buf)[i];
    ref = (function == reduceFunction::MAX) ? (*op_buf)[i] : (*op_buf)[i] *::size;
    EXPECT_TRUE(is_close(res, ref, FLOAT16RTOL, FLOAT16ATOL));
  }
}

TEST_F(ACCLTest, test_barrier) {
  if(!options.cyt_rdma){
    GTEST_SKIP() << "Barrier requires rendezvous support, currently only available on Coyote RDMA backend";
  }
  accl->barrier();
}

TEST_F(ACCLTest, test_perf_counter) {
  if(!options.hardware){
    GTEST_SKIP() << "Performance counter tests not run in simulation";
  }
  unsigned int count = options.count;
  auto op_buf = accl->create_buffer<float>(count, dataType::float32);
  auto res_buf = accl->create_buffer<float>(count, dataType::float32);
  random_array(op_buf->buffer(), count);

  test_debug("Reducing data...", options);
  ACCL::ACCLRequest* req = accl->nop(true);
  accl->wait(req);
  //check NOP call duration is between 100ns and 1us 
  bool cnt_in_range = (accl->get_duration(req) > 100) && (accl->get_duration(req) < 1000);
  EXPECT_TRUE(cnt_in_range);
}

INSTANTIATE_TEST_SUITE_P(reduction_tests, ACCLFuncTest, testing::Values(reduceFunction::SUM, reduceFunction::MAX));
INSTANTIATE_TEST_SUITE_P(rooted_tests, ACCLRootTest, testing::Range(0,::size));
INSTANTIATE_TEST_SUITE_P(rooted_reduction_tests, ACCLRootFuncTest, 
  testing::Combine(testing::Range(0,::size), testing::Values(reduceFunction::SUM, reduceFunction::MAX))
);
INSTANTIATE_TEST_SUITE_P(segmentation_tests, ACCLSegmentationTest, testing::Combine(testing::Values(1, 2), testing::Values(-1, 0, 1)));

options_t parse_options(int argc, char *argv[]) {
  TCLAP::CmdLine cmd("Test ACCL C++ driver");
  TCLAP::ValueArg<uint16_t> start_port_arg(
      "p", "start-port", "Start of range of ports usable for sim", false, 5500,
      "positive integer");
  cmd.add(start_port_arg);
  TCLAP::ValueArg<unsigned int> count_arg(
      "s", "count", "How many items per test", false, 16, "positive integer");
  cmd.add(count_arg);
  TCLAP::ValueArg<unsigned int> bufsize_arg("", "rxbuf-size",
                                            "How many KB per RX buffer", false,
                                            1, "positive integer");
  cmd.add(bufsize_arg);
  TCLAP::ValueArg<unsigned int> bufcount_arg("", "rxbuf-count",
                                            "How RX buffers", false,
                                            16, "positive integer");
  cmd.add(bufcount_arg);
  TCLAP::SwitchArg debug_arg("d", "debug", "Enable debug mode", cmd, false);
  TCLAP::SwitchArg hardware_arg("f", "hardware", "enable hardware mode", cmd,
                                false);
  TCLAP::SwitchArg axis3_arg("a", "axis3", "Use axis3 hardware setup", cmd,
                             false);
  TCLAP::SwitchArg udp_arg("u", "udp", "Use UDP hardware setup", cmd, false);
  TCLAP::SwitchArg tcp_arg("t", "tcp", "Use TCP hardware setup", cmd, false);
  TCLAP::SwitchArg cyt_tcp_arg("", "cyt_tcp", "Use Coyote TCP hardware setup", cmd, false);
  TCLAP::SwitchArg cyt_rdma_arg("", "cyt_rdma", "Use Coyote RDMA hardware setup", cmd, false);
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
  TCLAP::SwitchArg rsfec_arg("", "rsfec", "Enables RS-FEC in CMAC.", cmd, false);
  TCLAP::SwitchArg startemu_arg("", "startemu", "Start emulator processes locally.", cmd, false);
  TCLAP::SwitchArg bench_arg("", "benchmark", "Enables benchmarking.", cmd, false);
  TCLAP::ValueArg<std::string> csvfile_arg("", "csvfile",
                                          "Name of CSV file to be created for benchmark results",
                                          false, "", "string");
  TCLAP::ValueArg<unsigned int> max_eager_arg("", "max-eager-count",
                                            "Maximum byte count for eager mode", false,
                                            16*1024, "positive integer");
  cmd.add(max_eager_arg);
  try {
    cmd.parse(argc, argv);
    if (axis3_arg.getValue() + udp_arg.getValue() + tcp_arg.getValue() +
            cyt_rdma_arg.getValue() + cyt_tcp_arg.getValue() != 1) {
      throw std::runtime_error("Specify exactly one network backend out of axis3, "
                                "tcp, udp, cyt_tcp, or cyt_rdma modes.");
    }
  } catch (std::exception &e) {
    if (::rank == 0) {
      std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
  }

  options_t opts;
  opts.start_port = start_port_arg.getValue();
  opts.count = count_arg.getValue();
  opts.rxbuf_count = bufcount_arg.getValue();
  opts.rxbuf_size = bufsize_arg.getValue() * 1024; // convert to bytes
  opts.segment_size = std::min((unsigned)opts.rxbuf_size, (unsigned)4*1024*1024); //min of rxbuf_size and max_btt
  opts.debug = debug_arg.getValue();
  opts.hardware = hardware_arg.getValue();
  opts.axis3 = axis3_arg.getValue();
  opts.udp = udp_arg.getValue();
  opts.tcp = tcp_arg.getValue();
  opts.cyt_rdma = cyt_rdma_arg.getValue();
  opts.cyt_tcp = cyt_tcp_arg.getValue();
  opts.device_index = device_index_arg.getValue();
  opts.xclbin = xclbin_arg.getValue();
  opts.test_xrt_simulator = xrt_simulator_ready(opts);
  opts.config_file = config_arg.getValue();
  opts.rsfec = rsfec_arg.getValue();
  opts.startemu = startemu_arg.getValue();
  opts.benchmark = bench_arg.getValue();
  opts.csvfile = csvfile_arg.getValue();
  opts.max_eager_count = max_eager_arg.getValue();
  return opts;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &::size);

  //init google test with any arguments specific to it
  ::testing::InitGoogleTest(&argc, argv);

  //gather ACCL options for the test
  //NOTE: this has to come before the gtest environment is initialized
  options = parse_options(argc, argv);

  if(options.startemu){
    emulator_pid = start_emulator(options,::size,::rank);
    if(!emulator_is_running(emulator_pid)){
      std::cout << "Could not start emulator" << std::endl;
      return -1;
    }
  }

  // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
  ::testing::AddGlobalTestEnvironment(new TestEnvironment);

  bool fail = RUN_ALL_TESTS();
  std::cout << (fail ? "Some tests failed" : "All tests successful") << std::endl;

  if(options.startemu){
    kill_emulator(emulator_pid);
  }

  MPI_Finalize();
  return fail ? -1 : 0;
}
