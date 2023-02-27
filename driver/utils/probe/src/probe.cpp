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

ACCLProbe::ACCLProbe(xrt::device &device, xrt::kernel &probe, unsigned max_iter) : device(device), probe(probe), max_iter(max_iter), current_iter(0) {
    auto bankidx = probe.group_id(2); // Memory bank index for kernel argument 2
    buffer = xrt::bo(device, 16*4*max_iter, bankidx);
    run = xrt::run(probe);
}

ACCLProbe::~ACCLProbe(){}

void ACCLProbe::skip(unsigned niter){
    run.set_arg(0, false);
}

void ACCLProbe::arm(unsigned niter){
    run.set_arg(0, true);
    run.set_arg(1, (niter==0) ? 1 : niter);
    run.set_arg(2, buffer);
    current_iter = niter;
    if(niter == 0)
        run.start(xrt::autostart{niter});
    else
        run.start();
}

void ACCLProbe::disarm(){
    run.abort();
}

void ACCLProbe::read(){
    run.wait(1000);//wait for up to 1s in case the probe is still recording
    buffer.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
    for(unsigned i=0; i<current_iter; i++)
        durations.push_back(buffer.map<unsigned*>()[16*i+15]);
}

void ACCLProbe::dump(){
    for(unsigned duration : durations) 
        std::cout << "duration is " << duration << std::endl;
}
