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

#pragma once

#include <chrono>
#include <iostream>

using namespace std::chrono;

class Timer {
	private:
		steady_clock::time_point _start;
		steady_clock::time_point _end;	
		bool _started = false;
		bool _ended = false;
	public:
		Timer() {
			
		}
	
		void start() {
			_start = steady_clock::now();
			_started = true;
		}

		void end() {
			_end   = steady_clock::now();
			_ended = true;
		}

		unsigned long elapsed() {
			if(!_started || !_ended) {
				std::cerr << "Timer error. You forgot to call start or end or both" << std::endl;
				return 0;
			}
			return std::chrono::duration_cast<std::chrono::microseconds>(_end-_start).count();
		}
};
