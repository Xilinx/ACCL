/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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

#include <chrono>
#include <iostream>

/** @file timing.hpp */

namespace ACCL {
class Timer {
private:
  std::chrono::steady_clock::time_point _start;
  std::chrono::steady_clock::time_point _end;
  bool _started;
  bool _ended;

public:
  Timer() { reset(); }

  void start() {
    if (_started == true) {
      reset();
    }

    _start = std::chrono::steady_clock::now();
    _started = true;
  }

  void end() {
    if (_ended == true) {
      throw std::runtime_error("Timer already ended.");
    }

    if (_started == false) {
      throw std::runtime_error("Timer end called before start.");
    }

    _end = std::chrono::steady_clock::now();
    _ended = true;
  }

  void reset() {
    _started = false;
    _ended = false;
  }

  unsigned long elapsed() {
    if (!_started || !_ended) {
      throw std::runtime_error("You forgot to call start or end or both");
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start)
        .count();
  }
};
} // namespace ACCL
