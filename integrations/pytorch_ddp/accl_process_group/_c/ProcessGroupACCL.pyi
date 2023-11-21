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

from enum import Enum, unique
from torch.distributed.distributed_c10d import ProcessGroup, Store


@unique
class DataType(Enum):
   float16 = 2
   float32 = 3
   float64 = 4
   int32 = 5
   int64 = 6

@unique
class ACCLDesign(Enum):
   axis3x = 0
   tcp = 1
   udp = 2
   roce = 3
   cyt_tcp = 4
   cyt_rdma = 5


class Rank:
    def __init__(
        self: Rank,
        ip: str,
        port: int,
        session_id: int,
        max_segment_size: int) -> None: ...

    @property
    def ip(self: Rank) -> str: ...

    @property
    def port(self: Rank) -> int: ...

    @property
    def session_id(self: Rank) -> int: ...

    @property
    def max_segment_size(self: Rank) -> int: ...

    def __repr__(self: Rank) -> str: ...


class ProcessGroupACCL(ProcessGroup):
    def __init__(
        self: ProcessGroupACCL,
        store: Store,
        rank: int,
        size: int,
        ranks: list[Rank],
        simulator: bool,
        design: ACCLDesign,
        *,
        p2p_enabled: bool = False,
        compression: dict[DataType, DataType] = {},
        profiling_ranks: list[int] = [],
        profiling_timeout: float = 0.0,
        xclbin: str = '',
        device_index: int = 0,
        nbufs: int = 16,
        bufsize: int = 1024,
        rsfec: bool = False) -> None: ...

    def get_local_qp(self: ProcessGroupACCL, rank: int) -> list[int]: ...

    def set_remote_qp(self: ProcessGroupACCL, rank: int, qp: list[int]) -> None:
        ...

    def initialize(self: ProcessGroupACCL) -> None: ...

    @property
    def compression(self) -> dict[DataType, DataType]: ...

    @compression.setter
    def compression(self, value: dict[DataType, DataType]) -> None: ...
