from os import listdir, name
from os.path import isfile, join
from numpy import mean, std
import re
import csv
mypath ="C:\\Users\\danielep\\OneDrive - Xilinx, Inc\\XCCL_OFFLOAD_DOC\\execution_time\\bbd0da2"
csv_files       = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and f.find(".csv") != -1)]
result_file     = join(mypath, "res.csv")
csv_writer      = csv.writer( open(result_file, "w", newline=''), delimiter=",")
csv_writer.writerow(["bsize", "nruns", "bench_naccel", "segment_size", "collective_name", "avg_throughput", "std_throughput", "avg_execution_time", "std_execution_time"])
for f in csv_files:
    #name structure "bench_naccel{args.naccel}nruns{args.nruns}_bsize{args.bsize}_segment_size{args.segment_size}.csv"
    res = re.findall("bench_naccel(\d+)nruns(\d+)_bsize(\d+)_segment_size(\d+).csv", f)
    print(f)
    if len(res) == 0:
        continue

    bench_naccel = res[0][0]
    nruns        = res[0][1] 
    bsize        = res[0][2]
    segment_size = res[0][3]

    ##read first time to get collectives
    collectives = []
    csv_reader  = iter(csv.reader(open(f, "r", newline="") , delimiter=","))
    next(csv_reader)
    next(csv_reader)
    res = {}
    for row in csv_reader:
        print(row)

        collective_name = row[0]
        execution_time  = float(row[3])
        throughput      = float(row[4])
        if collective_name not in res.keys():
            res[collective_name] = {"throughput" : [], "execution_time" : []}
        res[collective_name]["throughput"].append(throughput)
        res[collective_name]["execution_time"].append(execution_time)
        

    for collective_name in res:
        avg_throughput      = mean(  res[collective_name]["throughput"])
        std_throughput      = std(   res[collective_name]["throughput"])
        avg_execution_time  = mean(  res[collective_name]["execution_time"])
        std_execution_time  = std(   res[collective_name]["execution_time"])

        csv_writer.writerow([bsize, nruns, bench_naccel, segment_size, collective_name, avg_throughput, std_throughput, avg_execution_time, std_execution_time])

    #csv_writer.writerow(["bsize[KB]", args.bsize, "segment[KB]", args.segment_size])
    #csv_writer.writerow(["collective", "size communicator" ,"size [KB]", "execution time [us]", "throughput[Gbps]", ])
