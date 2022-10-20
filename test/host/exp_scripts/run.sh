#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Script Dir: $SCRIPT_DIR"

echo "Compile"

cd $SCRIPT_DIR/../xrt && make

cd $SCRIPT_DIR

# server IDs (u55c)
SERVID=(9 10)
rm $SCRIPT_DIR/host
rm $SCRIPT_DIR/fpga
num_process=0
for servid in ${SERVID[@]}; do 
	# echo "alveo-u55c-$(printf "%02d" $servid)-mellanox-0:1" >>$SCRIPT_DIR/host
    echo "10.253.74.$(((servid-1) * 4+66))">>$SCRIPT_DIR/host
    echo "10.253.74.$(((servid-1) * 4+68))">>$SCRIPT_DIR/fpga
    num_process=$((num_process+1))
    hostlist+="alveo-u55c-$(printf "%02d" $servid) "
done

# Bitstream and argument configuration
HW_BENCH=0
USER_KERNEL=1

if [[ ($USER_KERNEL -eq 1) && ($HW_BENCH -eq 1)]]
then
    PREFIX="tcp_vadd_bench"
    ARG=" -d -f -t -z -k "
elif [[ ($USER_KERNEL -eq 0) && ($HW_BENCH -eq 1)]]
then
    PREFIX="tcp_vadd_bench"
    ARG=" -d -f -t -z "
elif [[ ($USER_KERNEL -eq 1) && ($HW_BENCH -eq 0)]]
then
    PREFIX="tcp_vadd"
    ARG=" -d -f -t -k "
elif [[ ($USER_KERNEL -eq 0) && ($HW_BENCH -eq 0)]]
then
    PREFIX="tcp"
    ARG=" -d -f -t "
else
    echo "NOT SUPPORTED CONFIGURATION!"
    exit
fi
XCLBIN=$SCRIPT_DIR/../../../test/hardware/link_${PREFIX}_eth_0_debug_none_xilinx_u55c_gen3x16_xdma_3_202210_1/ccl_offload.xclbin 

# Test Mode
#define ALL                 0
#define ACCL_COPY           1
#define ACCL_COMBINE        2
#define ACCL_SEND           3 
#define ACCL_RECV           4
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10
#define ACCL_REDUCE_SCATTER 11
#define ACCL_BARRIER        12

TEST_MODE=(3)
NUM_ELE_KILO=(2 4 8)
for np in `seq 2 $num_process`; do
    for test_mode in ${TEST_MODE[@]}; do 
        for num_ele_kilo in ${NUM_ELE_KILO[@]}; do 
            num_ele=$(((num_ele_kilo) * 1024))
            mpirun -n $np -f ./host --iface ens4f0 $SCRIPT_DIR/../xrt/bin/test $ARG -y $test_mode -c $num_ele -l $SCRIPT_DIR/fpga -x $XCLBIN &
            sleep 40
            parallel-ssh -H "$hostlist" "kill -9 \$(ps -aux | grep test | awk '{print \$2}')" 
            parallel-ssh -H "$hostlist" "/opt/xilinx/xrt/bin/xbutil reset --force --device 0000:c4:00.1"
        done
    done
done

#Post processing experiment log
