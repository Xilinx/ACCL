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
