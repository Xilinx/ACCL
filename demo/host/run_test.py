# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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

import argparse
import subprocess 
from os import write
from tqdm import tqdm
from itertools import product
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--xclbin"    ,    type=str , help="Accelerator image file (xclbin)"                ,default="../build/sequence_numbers/ccl_offload.xclbin")
    parser.add_argument("--board_idx" ,    type=int , help="Device (card) index"                            ,default=0)
    parser.add_argument("--nruns"    ,    type=int , help="How many times to run each test"               , default=1000)

    args = parser.parse_args()

    bsize   = [8*pow(10,i) for i in range(6)] #from 8 to 800KB
    sw_flag = [ True, False]

    for bsize,sw_flag in tqdm(list(product(bsize,sw_flag))):
        
        run_cmd =f"python test.py --xclbin {args.xclbin} --naccel 4 --device {args.board_idx} --nruns {args.nruns} --bsize {bsize} --nbufs 16 --all --bench " 
        run_cmd += ("--sw" if sw_flag else " ")
        print(run_cmd)
        p       = subprocess.run(run_cmd, shell=True, check=True, capture_output=True)
        stdout  = p.stdout.decode("utf-8")

        label_file="sw" if sw_flag else "hw"
        with open(f"log{bsize}{label_file}.txt", "w") as f:
            f.write(stdout)
