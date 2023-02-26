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

ACCLProbe::ACCLProbe(xrt::device &device, xrt::kernel &probe) : device(device), probe(probe) {
    auto bankidx = probe.group_id(2); // Memory bank index for kernel argument 2
    buffer = xrt::bo(device, 16*4, bankidx);
    run = xrt::run(probe);
}

ACCLProbe::~ACCLProbe(){
    run.abort();
}

void ACCLProbe::skip(unsigned niter){
    run.set_arg(0, false);
    run.set_arg(1, 1);
    run.set_arg(2, buffer);
    run.start(xrt::autostart{niter});
}

void ACCLProbe::arm(){
    run.set_arg(0, false);
    run.set_arg(1, 1);
    run.set_arg(2, buffer);
    run.start();
}

void ACCLProbe::disarm(){
    run.abort();
}

void ACCLProbe::read(){
    run.wait(1000);//wait for 1s
    buffer.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
    durations.push_back(buffer.map<unsigned*>()[15]);
}

void ACCLProbe::dump(){
    for(unsigned duration : durations) 
        std::cout << "duration is " << duration << std::endl;
}
