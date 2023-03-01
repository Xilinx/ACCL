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
#include <vector>
#include <string>
#include <fstream>

class ACCLProbe {
public:
  ACCLProbe(xrt::device &device, xrt::kernel &ip, std::string csvfile, unsigned max_iter=100);
  ~ACCLProbe();
  void arm(unsigned niter=1);
  void read(bool append=false);
  void flush();
  std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>> durations;

private:
  xrt::device device;
  xrt::kernel probe;
  xrt::bo buffer;
  xrt::run run;
  unsigned max_iter;
  unsigned current_iter;
  std::ofstream csvstream;
};
