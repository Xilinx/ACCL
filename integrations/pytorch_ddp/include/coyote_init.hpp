#pragma once
#include <vector>

#include <accl.hpp>

namespace coyote_init {
void setup_cyt_rdma(std::vector<fpga::ibvQpConn*> &ibvQpConn_vec,
                    std::vector<ACCL::rank_t> &ranks, int local_rank,
                    ACCL::CoyoteDevice &device);
void configure_cyt_rdma(std::vector<fpga::ibvQpConn*> &ibvQpConn_vec,
                        std::vector<ACCL::rank_t> &ranks, int local_rank);
} // namespace coyote_init
