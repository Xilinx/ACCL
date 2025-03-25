# ACCL: Alveo Collective Communication Library

[![Documentation Status](https://readthedocs.org/projects/accl/badge/?version=latest)](https://accl.readthedocs.io/en/latest/?badge=latest)
![Tests](https://github.com/Xilinx/ACCL/actions/workflows/build-and-test.yml/badge.svg)

ACCL is a Vitis kernel and associated XRT drivers which together provide MPI-like collectives for Xilinx FPGAs. ACCL is designed to enable compute kernels resident in FPGA fabric to communicate directly under host supervision but without requiring data movement between the FPGA and host. Instead, ACCL uses Vitis-compatible TCP and UDP stacks to connect FPGAs directly over Ethernet at up to 100 Gbps on Alveo cards.

ACCL currently supports Send/Recv and the following collectives:
* Broadcast
* Scatter
* Gather
* All-gather
* Reduce
* All-reduce
* Reduce-Scatter

## Installation
See [INSTALL.md](INSTALL.md) to learn how to build ACCL-enabled designs and interact with them from C++.
To use ACCL from Python, refer to [PyACCL](https://github.com/Xilinx/pyaccl).

## Citation
If you use our work or would like to cite it in your own, please cite one of our papers:

```
@INPROCEEDINGS{298689,
  author = {Zhenhao He and Dario Korolija and Yu Zhu and Benjamin Ramhorst and Tristan Laan and Lucian Petrica and Michaela Blott and Gustavo Alonso},
  title = {{ACCL+}: an {FPGA-Based} Collective Engine for Distributed Applications},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  year = {2024},
  isbn = {978-1-939133-40-3},
  address = {Santa Clara, CA},
  pages = {211--231},
  url = {https://www.usenix.org/conference/osdi24/presentation/he},
  publisher = {USENIX Association},
  month = jul
}
```
```
@INPROCEEDINGS{9651265,
  author={He, Zhenhao and Parravicini, Daniele and Petrica, Lucian and O’Brien, Kenneth and Alonso, Gustavo and Blott, Michaela},
  booktitle={2021 IEEE/ACM International Workshop on Heterogeneous High-performance Reconfigurable Computing (H2RC)},
  title={ACCL: FPGA-Accelerated Collectives over 100 Gbps TCP-IP},
  year={2021},
  pages={33-43},
  doi={10.1109/H2RC54759.2021.00009}}
```
