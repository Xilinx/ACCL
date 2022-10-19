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

#include "accl_hls.h"

#ifndef NUM_CTRL_STREAMS
#define NUM_CTRL_STREAMS 2
#endif

void client_arbiter(
	STREAM<command_word> &cmd_clients_0,
	STREAM<command_word> &ack_clients_0,
	STREAM<command_word> &cmd_clients_1,
	STREAM<command_word> &ack_clients_1,
	STREAM<command_word> &cmd_cclo,
	STREAM<command_word> &ack_cclo
);