SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Script Dir: $SCRIPT_DIR"

echo "Compile"

cd $SCRIPT_DIR/../xrt && make

cd $SCRIPT_DIR

# server IDs (u55c)
SERVID=(5 6 7 8)
rm $SCRIPT_DIR/host
rm $SCRIPT_DIR/fpga
num_process=0
for servid in ${SERVID[@]}; do 
	echo "alveo-u55c-$(printf "%02d" $servid)-mellanox-0:1" >>$SCRIPT_DIR/host
    echo "10.253.74.$(((servid-1) * 4+68))">>$SCRIPT_DIR/fpga
    num_process=$((num_process+1))
    hostlist+="alveo-u55c-$(printf "%02d" $servid) "
done

mpirun -n $num_process -f ./host --iface ens4f0 $SCRIPT_DIR/../xrt/bin/test -d -f -t -l $SCRIPT_DIR/fpga -x $SCRIPT_DIR/../../../test/hardware/link_tcp_eth_0_debug_none_xilinx_u55c_gen3x16_xdma_3_202210_1/ccl_offload.xclbin 

parallel-ssh -H "$hostlist" "/opt/xilinx/xrt/bin/xbutil reset --force --device 0000:c4:00.1"
