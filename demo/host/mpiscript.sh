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

source /opt/xilinx/xrt/setup.sh
source /opt/tools/Xilinx/Vitis/2020.2/.settings64-Vitis.sh
source /opt/tools/external/anaconda/bin/activate pynq-dask

#python test_mpi4py.py 
cd ~/XCCL_Offload_tcp+udp/demo/host
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 30 --bsize 1024  --send 
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 30 --bsize 1024  --bcast 
python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 10 --bsize 10  --send --bcast --scatter --gather --reduce --allgather --allreduce
#--reduce
#--scatter --gather 
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 1 --bsize 1024  --scatter
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 1 --bsize 1024  --gather
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 1 --bsize 1024  --allgather
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 1 --bsize 1024  --reduce
#python test_tcp_cmac_seq_mpi.py --xclbin ../build/single/ccl_offload.xclbin --device 0 --nruns 1 --bsize 1024  --allreduce