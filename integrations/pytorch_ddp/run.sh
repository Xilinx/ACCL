#!/bin/bash

#check working directory
if [[ $(pwd) != *pytorch_ddp ]]; then
	echo "ERROR: this script should only be run in the pytorch_ddp dir of the repo!"
	exit 1
fi

# state variables
mkdir -p "$(pwd)/accl_log"
# BUILD_DIR=../build
# point this to python venv, which has the relevant libraries installed
VENV_ACTIVATE=$(pwd)/venv/bin/activate
SETUP_SH=$(pwd)/setup.sh
SCRIPT=$(pwd)/test/test-generic.py
HOST_FILE=./accl_log/host
FPGA_FILE=./accl_log/fpga

#enter venv and run script
EXEC="bash -c \"source $VENV_ACTIVATE && source $SETUP_SH  && python $SCRIPT"
# EXEC="python $SCRIPT"


#---------------Setting up vars-------------
if [[ $ACCL_SIM -eq 1 ]]; then
    echo "Starting in simulator mode. Make sure to start the emulator beforehand"
    ARG="-s "

    if [[ -v ACCL_NP ]]; then
        NUM_PROCESS="$ACCL_NP"
    else
    	echo "Variable ACCL_NP not set. Enter num of processes:"
	read -a NUM_PROCESS
    fi

else
    echo "Starting in hw mode. Make sure to run flow_u55c beforehand."
    if [[ -v U55C_IDS ]]; then
	IFS=' ' read -r -a SERVID <<< "$U55C_IDS"
    else
	# read server ids from user
	echo "Variable U55C_IDS not set. Enter u55c machine ids (space separated):"
	read -a SERVID
    fi
    RANK_PORT="30501"
    # create ip files
    rm -f $HOST_FILE $FPGA_FILE
    NUM_PROCESS=0
    for ID in ${SERVID[@]}; do
	echo "10.253.74.$(((ID-1) * 4 + 66))">>$HOST_FILE
	echo "10.253.74.$(((ID-1) * 4 + 68))">>$FPGA_FILE
	NUM_PROCESS=$((NUM_PROCESS+1))
	HOST_LIST+="alveo-u55c-$(printf "%02d" $ID) "
	HOST_PORT_LIST+="alveo-u55c-$(printf "%02d" $ID):$RANK_PORT "
    done

    echo "HOST_LIST: ${HOST_LIST[*]}"

    #set master address
    MASTER_IP="10.253.74.$(((${SERVID[0]}-1) * 4 + 66))"
    MASTER_PORT="30501"

    echo "Master node set to: $MASTER_IP:$MASTER_PORT"

    MPI_ARGS="-f $HOST_FILE --iface ens4f0"
fi

ARG="$ARG -c cyt_rdma\""

#---------------Running it-------------

echo "Run command: $EXEC $ARG"

echo "Running with $NUM_PROCESS Processes"

rm -f $(pwd)/accl_log/rank*

C="mpirun -n $NUM_PROCESS $MPI_ARGS -outfile-pattern \"$(pwd)/accl_log/rank_%r_M_${MODE}_N_${N}_H_${H}_P_${P}_stdout\" -errfile-pattern \"$(pwd)/accl_log/rank_%r_M_${MODE}_N_${N}_H_${H}_P_${P}_stderr\" $EXEC $ARG"
# C="mpirun -n $NUM_PROCESS -f $HOST_FILE --iface ens4f0 $EXEC $ARG &"
echo $C

/bin/sh -c "$C"

if ! [[ $ACCL_SIM -eq 1 ]]; then
    SLEEPTIME=8
    echo "Sleep for $SLEEPTIMEs"
    sleep $SLEEPTIME
    parallel-ssh -H "$HOST_LIST" "killall -9 test-generic.py"
    parallel-ssh -H "$HOST_LIST" "dmesg | grep "fpga_tlb_miss_isr" >$(pwd)/accl_log/tlb_miss.log"
    # done

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
fi




