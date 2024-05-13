#!/bin/bash

echo "Enter ETHZ HACC Alveo U55C machine IDs (space separated, e.g. 4 5):"
read -a SERVID

echo "Enter path to ACCL .xclbin driver after bitstream generation:"
read -a XCLBIN_PATH

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HOST_FILE=$SCRIPT_DIR/host.txt
FPGA_FILE=$SCRIPT_DIR/fpga.json
rm $HOST_FILE $FPGA_FILE

# Obtain CPU (needed for launching MPI process) and FPGA (needed for ACCL-EasyNet) IPs
NP=0
for ID in ${SERVID[@]}; do 
	echo "10.253.74.$((($ID - 1) * 4 + 66))" >> $HOST_FILE
	fpgaip+="\"10.253.74.$((($ID - 1) * 4 + 68))\","
	hostlist+="alveo-u55c-$(printf "%02d" $servid) "
    NP=$((NP+1))
done
echo "{\"ips\": [${fpgaip::-1}]}" >> $FPGA_FILE

# Run application
mpirun -np $NP -iface ens4f0 -f $HOST_FILE $SCRIPT_DIR/build/bin/test -f -c $FPGA_FILE -x $XCLBIN_PATH &
sleep 30

# Kill process, clean-up IP files and reset device
rm $HOST_FILE $FPGA_FILE
parallel-ssh -H "$hostlist" "kill -9 \$(ps -aux | grep test | awk '{print \$2}')" 
parallel-ssh -H "$hostlist" "xbutil reset --force --device 0000:c4:00.1"

# /home/bramhorst/accl_vadd/test/refdesigns/link_tcp_xilinx_u55c_gen3x16_xdma_3_202210_1_1/ccl_offload.xclbin
# /home/bramhorst/accl_vadd/test/refdesigns/link_tcp_xilinx_u55c_gen3x16_xdma_3_202210_1_2/ccl_offload.xclbin