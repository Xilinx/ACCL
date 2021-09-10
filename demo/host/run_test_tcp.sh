#!/bin/sh

# /*******************************************************************************
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

#alveo3b 10.1.212.126
#alveo3c 10.1.212.127
#alveo4b 10.1.212.129
#alveo4c 10.1.212.130 
mpiexec --host 10.1.212.126,10.1.212.127,10.1.212.129,10.1.212.130 -- bash mpiscript.sh