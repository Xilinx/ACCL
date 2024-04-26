#!/bin/bash

#check working directory
if [[ $(pwd) != *pytorch_ddp/test ]]; then
	echo "ERROR: this script should only be run in the pytorch_ddp/test dir of the repo!"
	exit 1
fi

if [[ -v ACCL_SCRIPT ]]; then
    SCRIPT_NAME="$ACCL_SCRIPT"
else
    SCRIPT_NAME=test-generic.py
    echo "Variable ACCL_SCRIPT not set. Assuming $SCRIPT_NAME"
fi

# state variables
mkdir -p "$(pwd)/accl_log"
# BUILD_DIR=../build
# point this to python venv, which has the relevant libraries installed
VENV_ACTIVATE=$(pwd)/../venv/bin/activate
SETUP_SH=$(pwd)/../setup.sh
SCRIPT=$(pwd)/$SCRIPT_NAME
HOST_FILE=./accl_log/host
FPGA_FILE=./accl_log/fpga

#enter venv and run script
EXEC="bash -c \"source $VENV_ACTIVATE && source $SETUP_SH  && python $SCRIPT"
# EXEC="python $SCRIPT"


#---------------Setting up vars-------------
if [[ $ACCL_SIM -eq 1 ]]; then
    echo "Starting in simulator mode. Make sure to start the emulator beforehand"
    ARG="-s "

    ACCL_COMMS="udp"

    echo "assuming udp comms in simulator"

    if [[ -v ACCL_NP ]]; then
        NUM_PROCESS="$ACCL_NP"
    else
    	echo "Variable ACCL_NP not set. Enter num of processes:"
	read -a NUM_PROCESS
    fi

    MASTER_IP="localhost"
    MASTER_PORT="30505"

else
    echo "Starting in hw mode. Make sure to run flow_u55c beforehand."
    if [[ -v U55C_IDS ]]; then
	IFS=' ' read -r -a SERVID <<< "$U55C_IDS"
    else
	# read server ids from user
	echo "Variable U55C_IDS not set. Enter u55c machine ids (space separated):"
	read -a SERVID
    fi

    if ! [[ -v ACCL_COMMS ]]; then
        ACCL_COMMS="cyt_rdma"
	echo "Assuming cyt_rdma comms in hardware"
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
    MASTER_PORT="30505"

    echo "Master node set to: $MASTER_IP:$MASTER_PORT"

    MPI_ARGS="-f $HOST_FILE --iface ens4f0"
    # 09 and 10 have other interface names:
    # MPI_ARGS="-f $HOST_FILE --iface ens4"
fi

ARG="$ARG -c $ACCL_COMMS -i $HOST_FILE -f $FPGA_FILE -a $MASTER_IP -p $MASTER_PORT\""

#---------------Running it-------------

echo "Run command: $EXEC $ARG"

echo "Running with $NUM_PROCESS Processes"

rm -f $(pwd)/accl_log/rank*

# C="mpirun -n $NUM_PROCESS $MPI_ARGS -outfile-pattern \"$(pwd)/accl_log/rank_%r_stdout\" $EXEC $ARG &"
C="mpirun -n $NUM_PROCESS $MPI_ARGS -outfile-pattern \"$(pwd)/accl_log/rank_%r_stdout\" -errfile-pattern \"$(pwd)/accl_log/rank_%r_stderr\" $EXEC $ARG &"
# C="mpirun -n $NUM_PROCESS $MPI_ARGS $EXEC $ARG &"
echo $C

/bin/sh -c "$C"

if ! [[ -v SLEEPTIME ]]; then
    SLEEPTIME="16"
fi
echo "Sleeping for $SLEEPTIME"
sleep $SLEEPTIME

# if ! [[ $ACCL_SIM -eq 1 ]]; then
    # parallel-ssh -H "$HOST_LIST" "killall -9 $SCRIPT_NAME"
    # parallel-ssh -H "$HOST_LIST" "dmesg | grep "fpga_tlb_miss_isr" >$(pwd)/accl_log/tlb_miss.log"
# else
    # killall -9 $SCRIPT_NAME
    # dmesg | grep "fpga_tlb_miss_isr" >$(pwd)/accl_log/tlb_miss.log
# fi

# mkdir -p "$(pwd)/accl_results"
# # Loop through accl log files in the source directory and append to accl_results folder
# for source_log in "$(pwd)/accl"*.log; do
#     # Extract the log number from the source log file name (assuming the format is acclX.log)
#     log_number=$(basename "${source_log}" | sed 's/accl\([0-9]*\)\.log/\1/')
#     # Create the destination log file path
#     destination_log="$(pwd)/accl_results/accl${log_number}.log"
#     # Append the content of the source log to the destination log
#     cat "${source_log}" >> "${destination_log}"
#     # Remove the tmp log
#     rm ${source_log}
# done
