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
#include "cclo.hpp"
#include "constants.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <xrt/xrt_kernel.h>

/** @file acclrequest.hpp */

using namespace std::chrono_literals;

namespace ACCL {
/**
 * Generic Base Request class. Each type of device should derive an Request type
 * from this class, as they may have different ways to handle requests
 * 
 */
class BaseRequest {
private:
  std::mutex cv_mtx;
  std::condition_variable cv;
  std::atomic<operationStatus> status;
  val_t retcode;
  val_t duration_ncycles;

protected:
  // opaque reference to cclo
  void *cclo_ptr;

public:
  /**
   * Construct a new generic object
   * 
   * @param cclo Opaque reference to the main CCLO object
   */
  BaseRequest(void *cclo) : cclo_ptr(cclo) {
    status = operationStatus::QUEUED;
  }

  /**
   * Waits for a request to be completed
   *
   * @param timeout   Optional parameter specifying time to wait
   * @return true     The operation was completed
   * @return false    The operations didn't complete yet
   */
  bool wait(std::chrono::milliseconds timeout = 0ms) {
    std::unique_lock<std::mutex> lk(cv_mtx);
    bool ret = true;
    // Avoid waiting if operation already completed
    if (status.load() == operationStatus::COMPLETED)
      return ret;
    // Wait for a given timeout
    if (timeout > 0ms) {
      ret = cv.wait_for(lk, timeout, [&] {
        return status.load() == operationStatus::COMPLETED;
      });
    } else {
      // Wait undefinitely if no timeout set
      cv.wait(lk, [&] { return status.load() == operationStatus::COMPLETED; });
    }
    return ret;
  }

  /**
   * Notify waiting threads to check conditional variables
   * 
   */
  void notify() {
    std::lock_guard<std::mutex> lk(cv_mtx);
    cv.notify_all();
  }

  /**
   * Return the current status of this request
   * 
   * @return operationStatus Status of the request.
   */
  operationStatus get_status() {
    return status.load();
  }

  void set_status(operationStatus status) {
    this->status = status;
  }

  /**
   * Gets the value of the return code of the operation; used on internal checking.
   * 
   * @return val_t Return code of the operation
   */
  val_t get_retcode() const {
    return retcode;
  }

  /**
   * Sets the return code of the operation; used on internal checking.
   * 
   * @param retcode 
   */
  void set_retcode(val_t retcode) {
    this->retcode = retcode;
  }

  /**
   * Sets the duration of the operation; used for profiling.
   * 
   * @param retcode 
   */
  void set_duration(val_t duration) {
    this->duration_ncycles = duration;
  }

  /**
   * Gets the duration of the operation, in CCLO cycles.
   * 
   * @return val_t Duration of the operation
   */
  val_t get_duration() const {
    return duration_ncycles;
  }

  void *cclo() {
    return cclo_ptr;
  }
};

/**
 * Implementation of a thread-safe queue mechanism to manage requests
 *
 */
template <typename qtype> struct FPGAQueue {
  FPGAQueue() : qstatus(queueStatus::IDLE), unique_id(0) {}

  FPGAQueue &operator=(const FPGAQueue &other) {
    this->queue = other.queue;
    return *this;
  }

  ACCLRequest push(qtype item) {
    std::lock_guard<std::mutex> lk(queue_mtx);
    queue.push(item);
    unique_id++;
    return unique_id;
  }

  /**
   * Removes the front() item of the queue. Also sets the qstatus to idle
   *
   */
  void pop() {
    std::lock_guard<std::mutex> lk(queue_mtx);
    if (!queue.empty())
      queue.pop();
    if (qstatus == queueStatus::BUSY)
      qstatus = queueStatus::IDLE;
  }

  qtype front() {
    std::lock_guard<std::mutex> lk(queue_mtx);
    return queue.front();
  }

  bool empty() {
    std::lock_guard<std::mutex> lk(queue_mtx);
    return queue.empty();
  }

  /**
   * Tries to set the execution
   *
   * @return true  Execution was guaranteed to the caller
   * @return false  Execution was not guaranteed to the caller
   */
  bool run() {
    std::lock_guard<std::mutex> lk(queue_mtx);
    if (qstatus == queueStatus::IDLE && !queue.empty()) {
      qstatus = queueStatus::BUSY;
      return true;
    } else {
      return false;
    }
  }

private:
  queueStatus qstatus;
  std::queue<qtype> queue;
  std::mutex queue_mtx;
  std::atomic<ACCLRequest> unique_id;
};
} // namespace ACCL