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
declare -a arr=("send" "bcast" "scatter" "gather" "allgather" "reduce" "allreduce")
cd ~/ACCL/demo/host
ele_consec="1 2 4 8 16 32 64 128 256 512"
ele=(1024 2048 4096 8192 16384 32768)
numrun=20


for col in "${arr[@]}"
do

    python test_tcp.py --xclbin ../build/tcp_u280_debug/ccl_offload.xclbin --experiment measures --device 0 --nbufs 60 --nruns $numrun --segment_size 1024 --bsize $ele_consec --$col --use_tcp
    xbutil program -p ../build/tri/ccl_offload.xclbin

done


for col in "${arr[@]}"
do
    for i in "${ele[@]}" 
    do
        python test_tcp.py --xclbin ../build/tcp_u280_debug/ccl_offload.xclbin --experiment measures --device 0 --nbufs 60 --nruns $numrun --segment_size 1024 --bsize $i --$col --use_tcp
        xbutil program -p ../build/tri/ccl_offload.xclbin
    done
done
