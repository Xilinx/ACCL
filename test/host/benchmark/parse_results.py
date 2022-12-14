#!/usr/bin/env python3
# /*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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
# *******************************************************************************/

import json
import numpy
import math
import matplotlib.pyplot as plt
import statistics

from pathlib import Path
from typing import Dict, List, Tuple, Union


def _ideal(times: List[float], rtt: float) -> List[float]:
    return [time - rtt / 2 for time in times]


def _bw(times: List[float], size: int) -> List[float]:
    return [(size * 8) / time for time in times]


def _stats(data: List[float]) -> Tuple[float, float]:
        mean = statistics.mean(data)
        dev = statistics.stdev(data, mean)
        return mean, dev


class ResultsParser:
    def __init__(self, input_file: Path) -> None:
        with input_file.open('r') as input:
            self.data: dict = json.load(input)
        self.latencies: List[float] = self.data['latency']
        self.rtts: List[float] = self.data['rtt']
        self.sizes: List[int] = sorted(int(k) * 4 for k in self.data['throughput'])
        self.throughputs: Dict[int, List[float]] = \
            {int(k) * 4: v for k, v in self.data['throughput'].items()}
        self.allreduces: Dict[int, List[float]] = \
            {int(k) * 4: v for k, v in self.data['allreduce'].items()}

    @property
    def latency(self) -> Tuple[float, float]:
        return _stats(self.latencies)

    @property
    def rtt(self) -> Tuple[float, float]:
        return _stats(self.rtts)

    @property
    def send_recv(self) -> Dict[int, Tuple[float, float]]:
        return {size: _stats(self.throughputs[size])
                for size in self.sizes}

    @property
    def allreduce(self) -> Dict[int, Tuple[float, float]]:
        return {size: _stats(self.allreduces[size])
                for size in self.sizes}

    @property
    def send_recv_bw(self) -> Dict[int, Tuple[float, float]]:
        return {size: _stats(_bw(_ideal(self.throughputs[size], self.rtt[0] - self.rtt[1]), size))
                for size in self.sizes}

    @property
    def allreduce_bw(self) -> Dict[int, Tuple[float, float]]:
        return {size: _stats(_bw(_ideal(self.allreduces[size], self.rtt[0] - self.rtt[1]), size))
                for size in self.sizes}


def print_results(results: ResultsParser):
    def data_unit(size: Union[int, float], bw: bool = False) -> str:
        pre = 'bps' if bw else 'B'
        if size < 1e2:
            return f"{round(size)} {pre}"
        elif size < 1e4:
            return f"{size / 1e3:.2f} k{pre}"
        elif size < 1e5:
            return f"{size / 1e3:.1f} k{pre}"
        elif size < 1e7:
            return f"{size / 1e6:.2f} M{pre}"
        elif size < 1e8:
            return f"{size / 1e6:.1f} M{pre}"
        elif size < 1e10:
            return f"{size / 1e9:.2f} G{pre}"
        else:
            return f"{size / 1e9:.1f} G{pre}"

    def time_unit(time: float) -> str:
        if time >= 1e2:
            return f"{round(time):d} s"
        elif time >= 1e1:
            return f"{time:.1f} s"
        elif time >= 1e-1:
            return f"{time:.2f} s"
        elif time >= 1e-2:
            return f"{time * 1e3:.1f} ms"
        elif time >= 1e-4:
            return f"{time * 1e3:.2f} ms"
        elif time >= 1e-5:
            return f"{time * 1e6:.1f} μs"
        elif time >= 1e-7:
            return f"{time * 1e6:.2f} μs"
        elif time >= 1e-8:
            return f"{time * 1e9:.2f} ns"
        elif time >= 1e-10:
            return f"{time * 1e9:.2f} ns"
        else:
            return f"{time:.3e} s"

    print(f"Latency: {time_unit(results.latency[0])} ± "
          f"{time_unit(results.latency[1])}")
    print(f"RTT: {time_unit(results.rtt[0])} ± {time_unit(results.rtt[1])}")
    print("Throughput: ",
          {f"{data_unit(size)}":
           f"{data_unit(through[0], True)} ± {data_unit(through[1], True)}"
           for size, through in results.send_recv_bw.items()})
    print("Allreduce: ",
          {f"{data_unit(size)}": f"{data_unit(allred[0], True)} ± {data_unit(allred[1], True)}"
           for size, allred in results.allreduce_bw.items()})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse benchmark results')
    parser.add_argument('-i', '--input-file', type=Path,
                        default='results-1.json', metavar='JSON',
                        help="Input JSON file (defaults to '%(default)s')")
    parser.add_argument('-o', '--output-directory', type=Path,
                        default='results/', metavar='DIR',
                        help="Output directory (defaults to '%(default)s')")
    args = parser.parse_args()

    results = ResultsParser(args.input_file)
    print_results(results)
