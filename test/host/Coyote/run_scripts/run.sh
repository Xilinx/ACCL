#!/bin/bash

#check working directory
if [[ $(pwd) != */test/host/Coyote/run_scripts ]]; then
	echo "ERROR: this script should only be run in the /test/host/Coyote/run_scripts of the repo!"
	exit 1
fi

# state variables
BUILD_DIR=../build
EXEC=$BUILD_DIR/accl_on_coyote
HOST_FILE=./host
FPGA_FILE=./fpga

# read server ids from user
echo "Enter u55c machine ids (space separated):"
read -a SERVID

# create ip files
rm -f $HOST_FILE $FPGA_FILE
NUM_PROCESS=0
for ID in ${SERVID[@]}; do
	echo "10.253.74.$(((ID-1) * 4 + 66))">>$HOST_FILE
	echo "10.253.74.$(((ID-1) * 4 + 68))">>$FPGA_FILE
	NUM_PROCESS=$((NUM_PROCESS+1))
	HOST_LIST+="alveo-u55c-$(printf "%02d" $ID) "
done

# Test Mode
#define ALL                 0
#define ACCL_SEND           3 
#define ACCL_BCAST          5
#define ACCL_SCATTER        6
#define ACCL_GATHER         7
#define ACCL_REDUCE         8
#define ACCL_ALLGATHER      9
#define ACCL_ALLREDUCE      10

ARG=" -d -f -t" # debug, hardware, and tcp flags
TEST_MODE=(5) 
N_ELEMENTS_KB=(16)
HOST=(1)

echo "Run command: $EXEC $ARG -y $TEST_MODE -c 1024 -l $FPGA_FILE"

for NP in `seq $NUM_PROCESS $NUM_PROCESS`; do
	for MODE in ${TEST_MODE[@]}; do
		for N_ELE in ${N_ELEMENTS_KB}; do
			N=$((N_ELE*1024))
			echo "mpirun -n $NP -f $HOST_FILE --iface ens4 $EXEC $ARG -z $HOST -y $MODE -c $N -l $FPGA_FILE &"
			mpirun -prepend-rank -n $NP -f $HOST_FILE --iface ens4 $EXEC $ARG -z $HOST -y $MODE -c $N -l $FPGA_FILE &
			SLEEPTIME=$(((NP-2)*2 + 10))
			sleep $SLEEPTIME
			parallel-ssh -H "$HOST_LIST" "kill -9 \$(ps -aux | grep accl_on_coyote | awk '{print \$2}')"
		done
	done
done
