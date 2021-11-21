#!/bin/bash

handler()
{
    killall cclo_emu
}

world_size=$1
start_port=$2

trap handler SIGINT
for ((i=0; i<$world_size; i++))
do
    ./cclo_emu $world_size $i $start_port 2>/dev/null &
done

wait