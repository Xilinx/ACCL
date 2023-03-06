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
#include <cassert>


ACCLProbe::ACCLProbe(xrt::device &device, xrt::kernel &probe, std::string csvfile, unsigned max_iter) : device(device), probe(probe), max_iter(max_iter), current_iter(0) {
    auto bankidx = probe.group_id(2); // Memory bank index for kernel argument 2
    buffer = xrt::bo(device, 16*4*max_iter, bankidx);
    run = xrt::run(probe);
    if(!csvfile.empty()){
        // Open CSV file stream
        csvstream.open(csvfile, std::ios_base::out);
        // Push the header to it
        csvstream << "Opcode,Count,Root,Compression,Stream,Cycles" << std::endl;
    }
}

ACCLProbe::~ACCLProbe(){}

void ACCLProbe::arm(unsigned niter){
    assert(niter > 0 && niter <= max_iter);
    run.set_arg(0, true);
    run.set_arg(1, niter);
    run.set_arg(2, buffer);
    current_iter = niter;
    run.start();
}

void ACCLProbe::read(bool append){
    run.wait(1000);//wait for up to 1s in case the probe is still recording
    buffer.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
    auto tmp = buffer.map<unsigned*>();
    if(!append){
        durations.clear();
    }
    for(unsigned i=0; i<current_iter; i++)
        durations.push_back(std::make_tuple(tmp[16*i], tmp[16*i+1], tmp[16*i+3], tmp[16*i+7], tmp[16*i+8], tmp[16*i+15]));
}

void ACCLProbe::flush(){
    for(auto entry : durations){
        auto [ op , count, root, compression, stream, cycles ] = entry;
        csvstream << op << "," << count << "," << root << "," << compression << "," << stream << "," << cycles << std::endl;
    }
    durations.clear();
}
