# /*****************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *****************************************************************************/

from __future__ import annotations
from typing import Optional
from . import ProcessGroupACCL, Rank, DataType, ACCLDesign
import torch
import logging
from torch.distributed import Backend
from torch.distributed.distributed_c10d import ProcessGroup, Store


process_group: Optional[ProcessGroupACCL] = None


def create_process_group(
        ranks: list[Rank], design: ACCLDesign,
        *, nbufs: int = 16, bufsize: int = 1024,
        compression: Optional[dict[DataType, DataType]] = None,
        p2p_enabled: bool = False, profiling_ranks: Optional[list[int]] = None,
        profiling_timeout: float = 0.0, rsfec: bool = False,
        simulation: bool = False,
        initialize: bool = True) -> ProcessGroup:

    if compression is None:
        compression = {}
    else:
        # Copy compression since it will be used later in the lambda function
        compression = compression.copy()

    if profiling_ranks is None:
        profiling_ranks = []
    else:
        profiling_ranks = profiling_ranks.copy()

    def create_process_group_wrapper(store, rank, size, _timeout):
        global process_group
        if process_group is not None:
            raise RuntimeError("ACCL ProcessGroup already created, "
                               "can only create one.")

        # if simulation:
            #overwrite the design choice in simulation
            # design = ACCLDesign.udp
        
        pg = ProcessGroupACCL(store, rank, size, ranks, simulation, design,
                              bufsize=bufsize, rsfec=rsfec, nbufs=nbufs,
                              compression=compression,
                              p2p_enabled=p2p_enabled,
                              profiling_ranks=profiling_ranks,
                              profiling_timeout=profiling_timeout)

        process_group = pg
        if initialize:
            pg.initialize()

        return pg

    Backend.register_backend("ACCL", create_process_group_wrapper)

# def create_simulate_process_group(ranks: list[Rank], *,
#                                   nbufs: int = 16, udp: bool = False,
#                                   compression: Optional[dict[DataType,
#                                                              DataType]] = None,
#                                   bufsize: int = 1024,
#                                   initialize: bool = True) -> ProcessGroup:
#     if compression is None:
#         compression = {}
#     else:
#         # Copy compression since it will be used later in the lambda function
#         compression = compression.copy()

#     def create_process_group_wrapper(store, rank, size, _timeout):
#         global process_group
#         if process_group is not None:
#             raise RuntimeError("ACCL ProcessGroup already created, "
#                                "can only create one.")

#         design = ACCLDesign.udp if udp else ACCLDesign.tcp

#         pg = ProcessGroupACCL(store, rank, size, ranks, True, design,
#                               compression=compression, nbufs=nbufs,
#                               bufsize=bufsize)

#         process_group = pg
#         if initialize:
#             pg.initialize()

#         return pg

#     Backend.register_backend("ACCL", create_process_group_wrapper)

# def create_process_group_coyote(
#         ranks: list[Rank], design: ACCLDesign,
#         *, nbufs: int = 16, bufsize: int = 1024,
#         compression: Optional[dict[DataType, DataType]] = None,
#         p2p_enabled: bool = False, profiling_ranks: Optional[list[int]] = None,
#         profiling_timeout: float = 0.0, rsfec: bool = False,
#         initialize: bool = False) -> ProcessGroup:
#     if design != ACCLDesign.cyt_rdma and design != ACCLDesign.cyt_tcp:
#         raise RuntimeError(f"{design} is an incompatible design for coyote")

#     if compression is None:
#         compression = {}
#     else:
#         # Copy compression since it will be used later in the lambda function
#         compression = compression.copy()

#     if profiling_ranks is None:
#         profiling_ranks = []
#     else:
#         profiling_ranks = profiling_ranks.copy()

#     def create_process_group_wrapper(store, rank, size, _timeout):
#         global process_group
#         if process_group is not None:
#             raise RuntimeError("ACCL ProcessGroup already created, "
#                                "can only create one.")

#         pg = ProcessGroupACCL(store, rank, size, ranks, False, design,
#                               bufsize=bufsize, rsfec=rsfec, nbufs=nbufs,
#                               compression=compression,
#                               p2p_enabled=p2p_enabled,
#                               profiling_ranks=profiling_ranks,
#                               profiling_timeout=profiling_timeout)

#         process_group = pg
#         if initialize:
#             pg.initialize()

#         return pg

#     Backend.register_backend("ACCL", create_process_group_wrapper)

def initialize() -> None:
    if process_group is None:
        raise RuntimeError("Cannot initialize before ACCL ProcessGroup "
                           "is created.")
    process_group.initialize()

def get_local_qp(rank: int) -> list[int]:
    if process_group is None:
        raise RuntimeError("Cannot get local qp before ACCL ProcessGroup "
                           "is created.")
    return process_group.get_local_qp(rank)

def set_remote_qp(rank: int, qp: list[int]) -> None:
    if process_group is None:
        raise RuntimeError("Cannot set remote qp before ACCL ProcessGroup "
                           "is created.")
    return process_group.set_remote_qp(rank, qp)

def set_compression(compression: dict[DataType, DataType]):
    if process_group is None:
        raise RuntimeError("Cannot set compression before ACCL ProcessGroup "
                           "is initialized.")
    process_group.compression = compression


def get_compression() -> dict[DataType, DataType]:
    if process_group is None:
        raise RuntimeError("Cannot get compression before ACCL ProcessGroup "
                           "is initialized.")
    return process_group.compression
