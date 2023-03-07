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
import pandas as pd

from pathlib import Path
from typing import Dict, List, Tuple, Union
from enum import IntEnum

class Opcode(IntEnum):
    CONFIG         = 0
    COPY           = 1
    COMBINE        = 2
    SEND           = 3 
    RECV           = 4
    BCAST          = 5
    SCATTER        = 6
    GATHER         = 7
    REDUCE         = 8
    ALLGATHER      = 9
    ALLREDUCE      = 10
    REDUCE_SCATTER = 11

# TODOs:
# - parse benchmark CSV(s) and plot duration vs message size, including the stddev
# - calculate bandwidth efficiency compared to ideal theoretical for each collective
#    - script takes world size and local rank as input; operation root is in the CSV
#    - one model per each primitive collective; model may differ based on where root is
#    - script takes frequency as input to convert cycles to microseconds

def ideal_duration(opcode, world_size, msg_size_bits, fhz, rtt_s, bw_bps):
    if(opcode == Opcode.SEND):
        duration = msg_size_bits/bw_bps
    elif(opcode == Opcode.BCAST):
        duration = (world_size-1)*(msg_size_bits/bw_bps)
    elif(opcode == Opcode.SCATTER):
        duration = ((msg_size_bits/world_size)/bw_bps)
    elif(opcode == Opcode.GATHER):
        duration = world_size*((rtt_s/2)+msg_size_bits/bw_bps)
    elif(opcode == Opcode.REDUCE):
        duration = world_size*((rtt_s/2)+msg_size_bits/bw_bps)
    elif(opcode == Opcode.ALLGATHER):
        duration = (world_size-1)*((rtt_s/2)+(msg_size_bits/world_size)/bw_bps)
    elif(opcode == Opcode.ALLREDUCE):
        duration = 2*(world_size-1)*((rtt_s/2)+(msg_size_bits/world_size)/bw_bps)
    elif(opcode == Opcode.REDUCE_SCATTER):
        duration = (world_size-1)*((rtt_s/2)+(msg_size_bits/world_size)/bw_bps)
    else:
        duration = 0
    return duration

class ResultsParser:
    def __init__(self, input_file, fmhz, size, rtt_ns, bw_gbps) -> None:
        results = pd.read_csv(input_file)
        results['duration_us'] = results['Cycles'] / fmhz
        results['msg_size_bytes'] = results['Count'] * 4
        self.op_results = {}
        for opcode in Opcode:
            if opcode == Opcode.CONFIG: continue
            if opcode == Opcode.COPY: continue
            if opcode == Opcode.COMBINE: continue
            if opcode == Opcode.RECV: continue
            tmp_df = results[results['Opcode'] == opcode].groupby('msg_size_bytes').agg({'duration_us':['mean', 'std']})
            tmp_df.columns = ['duration_us_mean', 'duration_us_stddev']
            tmp_df = tmp_df.reset_index()
            tmp_df['ideal_duration_us'] = tmp_df.apply(lambda x : ideal_duration(opcode, size, x.msg_size_bytes*8, fmhz*(10**6), rtt_ns/(10**9), bw_gbps*(10**9))*(10**6), axis=1)
            tmp_df['duration_ratio'] = tmp_df['duration_us_mean'] / tmp_df['ideal_duration_us']
            self.op_results[opcode] = tmp_df

    def print(self, path):
        with open(path+'/results.md', 'w') as f:
            for opcode in Opcode:
                try:
                    tmp_str = '\n\n' + opcode.name + '\n===========================\n\n'
                    tmp_str += self.op_results[opcode].to_markdown(index=False, floatfmt=(".0f", ".2f", ".2f", ".2f", ".0f"))
                    f.write(tmp_str)
                except:
                    continue

    def plot(self, path):
        for opcode in Opcode:
            try:
                self.op_results[opcode].plot(legend='', x='msg_size_bytes', y='duration_us_mean', yerr='duration_us_stddev', loglog=True)
            except:
                continue
            plt.xlabel('Message size (bytes)')
            plt.ylabel('Duration (us)')
            plt.title(opcode.name)
            plt.savefig(path+'/plot_'+str(opcode.name)+'.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse benchmark results')
    parser.add_argument('-i', '--input-file', type=str, required = True, help="Input CSV file")
    parser.add_argument('-o', '--output-directory', type=str, default='.', 
                        help="Output directory (defaults to '%(default)s')")
    parser.add_argument('-l', '--rtt_ns', type=int, default=500,
                        help="Round trip latency, in nanoseconds (defaults to '%(default)s')")
    parser.add_argument('-f', '--fmhz', type=int, default=250,
                        help="Operating frequency, in megahertz (defaults to '%(default)s')")
    parser.add_argument('-s', '--size', type=int, required=True, help="World size")
    parser.add_argument('-b', '--bw_gbps', type=int, default=100, help="Network bandwidth, in Gbps (defaults to '%(default)s')")
    parser.add_argument('-p', '--plot', action='store_true', help="Enable plotting")
    args = parser.parse_args()

    results = ResultsParser(args.input_file, args.fmhz, args.size, args.rtt_ns, args.bw_gbps)
    if args.plot:
        results.plot(args.output_directory)
        results.print(args.output_directory)
