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
# *******************************************************************************/

#include <iostream>

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_aie.h"


using namespace std;

int main(void) {


unsigned int dev_index = 0;
auto device = xrt::device(dev_index);

auto xclbin_uuid = device.load_xclbin("binary_container_1.xclbin");

auto krnl = xrt::kernel(device, xclbin_uuid, "krnl_vadd");


auto group_id_0 = 0;//krnl.group_id(0);
auto group_id_1 = 0;//krnl.group_id(1);

cout << group_id_0 << endl;
cout << group_id_1 << endl;

auto buffer_size_in_bytes = 1024 << 5;

try {
auto ah_buffer = xrt::bo(device, buffer_size_in_bytes, xrt::bo::flags::normal,  group_id_0);

auto bh_buffer = xrt::bo(device, buffer_size_in_bytes, xrt::bo::flags::normal, group_id_1);
	auto hostmap = ah_buffer.map<int*>();
	for(int i=0; i<buffer_size_in_bytes/sizeof(int); i++) {
		cout << hostmap[i] << endl;
	}

} catch (const std::exception& e) {
	cerr << "Error: "<< e.what() <<endl;
}



	return 0;
}
