/*******************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices Inc.
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

#include <probe.hpp>
#include <iostream>

namespace ACCL {
Probe::Probe(xrt::device &device, xrt::ip &probe) : probe(probe), device(device) {
    auto bankidx = probe.group_id(4); // Memory bank index for kernel argument 4
    buffer = xrt::bo(device, 16*4, bankidx);
    run = xrt::run(probe);
}

void Probe::skip(){
    run.set_arg(4, buffer);
    run.set_arg(5, false);
    run.set_arg(6, 1);
    run.start();
}

void Probe::arm(){
    run.set_arg(4, buffer);
    run.set_arg(5, false);
    run.set_arg(6, 1);
    run.start();
}

void Probe::read(){
    run.wait();
    buffer.sync_from_device();
    durations.push_back(buffer[15]);
}

void Probe::dump(){
    for(unsigned duration : durations) 
        std::cout << "duration is " << duration << std::endl;
}

} // namespace ACCL
