#!/bin/bash

#check working directory
if [[ $(pwd) != */test/host/Coyote/run_scripts ]]; then
	echo "ERROR: this script should only be run in the /test/host/Coyote/run_scripts of the repo!"
	exit 1
fi

# state variables
mkdir -p "$(pwd)/accl_log"
BUILD_DIR=../build
EXEC=$BUILD_DIR/accl_on_coyote
HOST_FILE=./accl_log/host
FPGA_FILE=./accl_log/fpga

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
#define ACCL_BARRIER 		12

ARG=" -d -f -r" # debug, hardware, and tcp/rdma flags
TEST_MODE=(10) 
N_ELEMENTS=(512) # 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
NRUN=(1) # number of runs
HOST=(1)
PROTOC=(1) # eager=0, rendezevous=1

echo "Run command: $EXEC $ARG -y $TEST_MODE -c 1024 -l $FPGA_FILE"

rm -f $(pwd)/accl_log/rank*

for NP in `seq 4 $NUM_PROCESS`; do
	for MODE in ${TEST_MODE[@]}; do
		for N_ELE in ${N_ELEMENTS[@]}; do
			for H in ${HOST[@]}; do
				for P in ${PROTOC[@]}; do
					N=$N_ELE
					echo "mpirun -n $NP -f $HOST_FILE --iface ens4 $EXEC $ARG -z $H -y $MODE -c $N -l $FPGA_FILE -p $P -n $NRUN &"
					mpirun -n $NP -f $HOST_FILE --iface ens4f0 -outfile-pattern "./accl_log/rank_%r_M_${MODE}_N_${N}_H_${H}_P_${P}_stdout" -errfile-pattern "./accl_log/rank_%r_M_${MODE}_N_${N}_H_${H}_P_${P}_stdout" $EXEC $ARG -z $H -y $MODE -c $N -l $FPGA_FILE -p $P -n $NRUN &
					SLEEPTIME=2
					sleep $SLEEPTIME
					parallel-ssh -H "$HOST_LIST" "kill -9 \$(ps -aux | grep accl_on_coyote | awk '{print \$2}')"
					parallel-ssh -H "$HOST_LIST" "dmesg | grep "fpga_tlb_miss_isr" >$(pwd)/accl_log/tlb_miss.log"
				done
			done
		done
	done
done

mkdir -p "$(pwd)/accl_results"
# Loop through accl log files in the source directory and append to accl_results folder
for source_log in "$(pwd)/accl"*.log; do
    # Extract the log number from the source log file name (assuming the format is acclX.log)
    log_number=$(basename "${source_log}" | sed 's/accl\([0-9]*\)\.log/\1/')
    # Create the destination log file path
    destination_log="$(pwd)/accl_results/accl${log_number}.log"
    # Append the content of the source log to the destination log
    cat "${source_log}" >> "${destination_log}"
	# Remove the tmp log
	rm ${source_log}
done
