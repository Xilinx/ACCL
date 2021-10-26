## Test your xclbin
To test your `.xclbin` on XACC do the following:
1. copy on XACC some of the ACCL directories maintaining the same structure as in the repo.
    ```
    ssh alveo0
    mkdir ACCL
    mkdir -p ACCL/demo/build/tcp_u280
    mkdir -p ACCL/demo/host/measurements/accl
    exit
    scp -r ACCL/demo/build/link*/ccl_offload.* alveo0:~/ACCL/demo/build/tcp_u280
    scp -r ACCL/driver alveo0:~/ACCL/
    scp -r ACCL/demo/host alveo0:~/ACCL/demo/
    ```
1.  login on one XACC node and go in the ACCL/demo/host
1.  modify ``run_test_tcp.sh`` to target a number of XACC

    ```
    #alveo3b 10.1.212.126
    #alveo3c 10.1.212.127
    #alveo4b 10.1.212.129
    #alveo4c 10.1.212.130 
    mpiexec --host 10.1.212.123,10.1.212.127,10.1.212.129,10.1.212.130 -- bash mpiscript.sh
    ```

1.  login to those hosts copying your ssh id 

    ```
    ssh-copy-id 10.1.212.123
    ssh-copy-id 10.1.212.127
    ssh-copy-id 10.1.212.129
    ssh-copy-id 10.1.212.130
    ```

1. when you run ``run_test_tcp.sh`` it will execute ``mpiscript.sh`` that will recall the python test script with the parameters. Example of ``mpiscript.sh``

    ```
    source /opt/xilinx/xrt/setup.sh
    source /opt/tools/Xilinx/Vitis/2020.2/.settings64-Vitis.sh
    source /opt/tools/external/anaconda/bin/activate pynq-dask

    #python test_mpi4py.py 
    cd ~/ACCL/demo/host
    python test_tcp_cmac_seq_mpi.py --xclbin ../build/tcp_u280_debug/ccl_offload.xclbin --device 0 --nruns 10 --segment_size 1024 --bsize 512 1024 2048 --send --bcast --gather --allgather --allreduce --reduce  

    ```

1. the ``test_tcp_cmac_seq_mpi.py`` will create a .``csv`` under ``./measurements/accl`` that logs the execution time for the collectives


