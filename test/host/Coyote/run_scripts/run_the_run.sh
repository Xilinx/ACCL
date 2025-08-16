#!/bin/bash
# Paths (adjust these if needed)
UTIL_PATH="/mnt/scratch/jooertli/ACCL_CYT_V2_EGR_HOST/test/refdesigns/Coyote/util"
BITSTREAM_PATH="/mnt/scratch/jooertli/ACCL_CYT_V2_EGR_HOST/test/refdesigns/coyote_build_RDMA_u55c_norrsp_nocomp_opt_tlb2_eager_host/bitstreams/cyt_top.bit"
DRIVER_PATH="/mnt/scratch/jooertli/ACCL_CYT_V2_EGR_HOST/test/refdesigns/Coyote/driver/build/coyote_driver.ko"

# Get FPGA numbers from the user for programming
read -p "Enter FPGA numbers (separated by space, e.g., '1 2'): " FPGA_INPUTS

NRUN=10
N_ELEMENTS=(4096 16384 65536 262144 524288) #4096 16384 65536 262144 524288 1048576
for N_ELE in ${N_ELEMENTS[@]}; do
    for (( run=1; run<=NRUN; run++ )); do
        echo "Run #$run for N_ELEMENTS=$N_ELE"
        # Run the programming script
        echo "Programming FPGAs: $FPGA_INPUTS ..."
        bash "$UTIL_PATH/program_hacc_remote.sh" "$BITSTREAM_PATH" "$DRIVER_PATH" <<< "$FPGA_INPUTS"
        SLEEPTIME=6
		sleep $SLEEPTIME
        # Run run_eval.sh with current N_ELE and same FPGA_INPUTS
        echo "Now running run_eval.sh with N_ELEMENTS=$N_ELE (run #$run)..."
        bash "$(dirname "$0")/run_eval.sh" "$N_ELE" <<< "$FPGA_INPUTS"
        SLEEPTIME=3
		sleep $SLEEPTIME
    done
done
echo "Done!"