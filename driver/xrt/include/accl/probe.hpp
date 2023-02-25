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

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace ACCL {

class Probe {
public:
  Probe(xrt::device &device, xrt::ip &probe);

void Probe::skip();
void Probe::arm();
void Probe::read();
void Probe::dump();

private:
  xrt::device device;
  xrt::ip probe;
  xrt::bo buffer;
  xrt::run run;
  vector<unsigned> durations;

};

} // namespace ACCL
