vec=(1 2 16 32 128 256 1024 2048 4096 8192 16384 32768 65536 131072 196608 262144 327680 393216 458752 524288 589824 655360 720896 786432 851968 917504 983040 1048576 1114112 1179648 1245184 1310720 1376256 1441792 1507328 1572864 1638400 1703936 1769472 1835008 1900544 1966080 2031616 2097152)

#TEST GPU buffers with P2P (HW time)

printf "gpu_to_fpga_to_gpu, fpga_to_gpu, gpu_to_fpga\n" > output_gpu_p2p.csv

for count in ${vec[@]}
do
    ./accl_on_coyote -y 1 -f -r -l fpgaIP -n $count -o output_gpu_p2p.csv -d
done

#TEST GPU buffers without P2P (SW time)

printf "gpu_to_fpga_to_gpu, fpga_to_gpu, gpu_to_fpga\n" > output_gpu_no_p2p.csv

for count in ${vec[@]}
do
    ./accl_on_coyote -y 15 -f -r -l fpgaIP -n $count -o output_gpu_no_p2p.csv -d
done

#TEST HOST buffers (HW time)

printf "host_to_fpga_to_host, fpga_to_host, host_to_fpga\n" > output_host.csv

for count in ${vec[@]}
do
    ./accl_on_coyote -y 14 -f -r -l fpgaIP -n $count -o output_host.csv -d
done

#TEST FPGA register mapping
 ./accl_on_coyote -y 16  -f -r -l fpgaIP -n 200 -o output.csv -d 
