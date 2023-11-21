This is a seperate repo based on https://github.com/Xilinx/ACCL/tree/pytorch_ddp

## Build

From within a virtual environment, do:

```sh
PYTORCH_ROCM_ARCH=gfx906 ROCM_HOME=/opt/rocm ACCL_DEBUG=0 ./install.py --rocm
```

This compiles PyTorch with the C++ ABI (for compatibility with ACCL) and with ROCM support for MI100 (gfx906),
then runs pip install in this folder with correct flags.

## Run tests
To run tests, source `setup.sh` and then run Python tests with `mpirun`. DO NOT RUN PYTHON IN THE ROOT DIRECTORY!
