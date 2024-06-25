This is a plugin for PyTorch that provides ACCL as a distributed backend for PyTorch. The plugin supports GPU tensors,
and can do P2P transfers between GPU and FPGA.

## Build
The PyTorch DDP plugin for ACCL requires a build of PyTorch that is compiled with the Cxx11 ABI. Unfortunately,
PyTorch only provides pre-Cxx11 ABI builds over pip. We provide an install script, which first builds PyTorch v1.12
with the correct compilation settings and then builds the plugin. To prevent overwriting the system PyTorch install,
we recommend installing the plugin into a virtual environment. To create a
virtual environment, run the following:
```bash
python3 -m venv venv
source venv/bin/activate
```

<details><summary>Installation without GPU support</summary>
  To install the plugin without GPU support, simply run the following from within the venv:

  ```bash
  ./install.py
  ```
</details>
<details><summary>Installation with CUDA support</summary>
  To install the plugin with Nvidia GPU support, run the following from within the venv:

  ```bash
  ./install.py --cuda
  ```
</details>

<details><summary>Installation with ROCm support</summary>
  To install the plugin with AMD GPU support, run the following from within the venv:

  ```bash
  ROCM_HOME=/opt/rocm ./install.py --rocm
  ```
  You can also manually specify your compute architecture using the `PYTORCH_ROCM_ARCH` environment variable. For
  example, to build the plugin for the AMD Instinct MI100, run the following:

  ```bash
  PYTORCH_ROCM_ARCH=gfx906 ROCM_HOME=/opt/rocm ./install.py --rocm
  ```
</details>

## Running the plugin

Make sure to source the `setup.sh` script in this directory to load the ACCL plugin before starting a Python script.
Example usage can be found in the various test files under [`test/`](test).

Do make sure not to run python from within the root directory of `pytorch_ddp`, because Python will try to import the
local incomplete [`accl_process_group/`](accl_process_group) folder instead of the actual installation.

The provided `test/run.sh` will launch a testscript via mpirun

## Setup overview

- The whole Processgroup is wrapped in OpenMPI, which is used for initialization
- You can use the OpenMPI implementation of certain collectives using the "sidestep" flags in the ProcessGroupACCL.cpp
- Recompilation using `./install` or `pip install .` can be very slow, you can run `python setup.py build_ext --inplace` and then copy the binary or other files directly. `cp accl_process_group/_c/ProcessGroupACCL.cpython-38-x86_64-linux-gnu.so ~/.local/lib/python3.8/site-packages/accl_process_group/_c/`
- The `install.py` script will not reinstall the driver in case of ACCL updates. You will need to rebuild it yourself
- Set `ACCL_DEBUG=1` if you want more output(also set during build). Stdout is sometimes not complete(in simulator), so best log most things in stderr
- The runscript currently just outputs the command to be run(better not use the `&` at the end), which you then run manually. This is because I had bad experiences with the missing output(maybe coinciding with issues mentioned above) and termination on multiple machines, but should also work if you comment the `exit 0` and the `&` at the end of mpirun out. Don't forget, that you should still run the script to clear log files.
- ACCL only supports sizes up to 4MB, If you give it tensors of higher sizes, the PG will try to segment it in first dim. Not all collectives correctly handle multi-dimensional tensors yet. 
- Setting up the simulator with 4MB takes long, better set it lower for quick tests.
- You can init the process group as if it were udp and run on a `cyt_rdma` simulator
- There is no reason to not support the rdma + SIM initialization. It just hasn't been implemented yet. Certain case-splits assume no-sim if cyt_rdma is given...

### How to install torchvision

- install torch using the script
- clone vision, go to the fitting version v0.16.0
- clone libpng, configure with prefix set to local directory
- add the bin to the path
- not sure if needed: supply the path of the library and include to torchvision as in their development doc
- disable the version check in torchvision setup.py, because it doesn't correctly parse the version.
- run vision setup.py with debug, include, library and use png flags

### Tests available
Check `test/run.sh` for ACCL_SCRIPT examples

- `test-generic.py` tests everything in isolation + a small dual layer model learning a linear function
- `test-mnist.py` should be able to be run non-distributed as well(check arguments)
- `test-imagenet.py` does finetuning of Resnet50 according to: <https://docs.ray.io/en/latest/train/examples/pytorch/pytorch_resnet_finetune.html> and should alse be able to be run non-distributed
- For DLRM you will need to use a small fork of the DLRM-repo with ACCL-support hosted at <https://gitlab.ethz.ch/lawirz/dlrm>. It contains a `run.sh`
