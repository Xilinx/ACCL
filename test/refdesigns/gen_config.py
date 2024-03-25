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

import argparse

parser = argparse.ArgumentParser(description='Generate ACCL linker config file')
parser.add_argument('-b', '--board', type=str, choices=['u50', 'u55c', 'u200', 'u250', 'u280'], default="u55c", help="Board name")
parser.add_argument('-p', '--poe', type=str, choices=['udp', 'tcp', 'axis3x'], default="tcp", help="Type of POE")
parser.add_argument('-o', '--outfile', type=str, default="link_config.ini", help="Name of generated config file")
parser.add_argument('--ethif', type=int, default=0, choices=[0, 1], help="Which Ethernet port to use, on cards which have two")
parser.add_argument('--vadd', action='store_true', help="Connect a Vadd kernel to the CCLO(s)")
parser.add_argument('--host', action='store_true', help="Connect to host memory - this implies use of CCLO configured for external DMA")
parser.add_argument('--hwemu', action='store_true', help="Replace CMAC with IPC AXIS Master/Slave for emulation")
parser.add_argument('--chipscope', action='store_true', help="Add Chipscope debug to CCLO streams")
args = parser.parse_args()

if args.board == "u50" and args.ethif != 0:
    raise "U50 has a single Ethernet port"

if args.host:
    if args.board == "u280" or args.board == "u50":
        raise "Host memory only supported on U55C/U200/U250"

if args.poe == "axis3x":
    args.axis3x = True
    args.poe = "tcp_dummy"
else:
    args.axis3x = False

num_cclo = 3 if args.axis3x else 1

# Kernels Instantiation
if args.hwemu and not args.axis3x:
    cmac_instantiation = "nk=sim_ipc_axis_master_512:1:gt_master\nnk=sim_ipc_axis_slave_512:1:gt_slave"
elif args.axis3x:
    cmac_instantiation = ""
elif args.poe == "tcp":
    cmac_instantiation = "nk=cmac_krnl:1:cmac"
else:
    cmac_instantiation = "nk=cmac_{eth_intf}:1:cmac".format(eth_intf=args.ethif)

if args.axis3x:
    poe_instantiation = "nk=network_krnl:3:"
    for i in range(num_cclo):
        endch = "" if i == num_cclo-1 else "."
        poe_instantiation += "poe_{inst_nr}".format(inst_nr=i) + endch
elif args.poe == "tcp":
    poe_instantiation = "nk=network_krnl:1:poe_0\n"
    poe_instantiation += "nk=tcp_session_handler:1:session_handler_0"
elif args.poe == "udp":
    poe_instantiation = "nk=networklayer:1:poe_0"
else:
    poe_instantiation = "nk=HiveNet_kernel_0:1:poe_0"

cclo_instantiation = "nk=ccl_offload:{num_inst}:".format(num_inst=num_cclo)
arb_instantiation = "nk=client_arbiter:{num_inst}:".format(num_inst=num_cclo)
hc_instantiation = "nk=hostctrl:{num_inst}:".format(num_inst=2*num_cclo)
reduce_instantiation = "nk=reduce_ops:{num_inst}:".format(num_inst=num_cclo)
cast_instantiation = "nk=hp_compression:{num_inst}:".format(num_inst=3*num_cclo)
extdma_instantiation = "nk=external_dma:{num_inst}:".format(num_inst=2*num_cclo)

for i in range(num_cclo):
    endch = "" if i == num_cclo-1 else "."
    cclo_instantiation += "ccl_offload_{inst_nr}".format(inst_nr=i) + endch
    arb_instantiation += "arb_{inst_nr}".format(inst_nr=i) + endch
    hc_instantiation += "hostctrl_{inst_nr}_0.hostctrl_{inst_nr}_1".format(inst_nr=i) + endch
    reduce_instantiation += "arith_{inst_nr}".format(inst_nr=i) + endch
    cast_instantiation += "compression_{inst_nr}_0.compression_{inst_nr}_1.compression_{inst_nr}_2".format(inst_nr=i) + endch
    extdma_instantiation += "extdma_{num_inst}_0.extdma_{num_inst}_1".format(num_inst=i) + endch

if args.axis3x:
    if args.vadd:
        loopback_instantiation = ""
    else:
        loopback_instantiation = "nk=loopback:3:lb_user_krnl_0.lb_user_krnl_1.lb_user_krnl_2"
elif args.poe == "tcp":
    if not args.vadd:
        loopback_instantiation = "nk=loopback:3:lb_user_krnl_0.lb_udp_txrx.lb_udp_meta"
    else:
        loopback_instantiation = "nk=loopback:2:lb_udp_txrx.lb_udp_meta"
elif not args.vadd:
    loopback_instantiation = "nk=loopback:1:lb_user_krnl_0"
else:
    loopback_instantiation = ""

if args.vadd:
    vadd_instantiation = "nk=vadd_put:{inst_nr}:".format(inst_nr=num_cclo)
    for i in range(num_cclo):
        endch = "" if i == num_cclo-1 else "."
        vadd_instantiation += "vadd_{inst_nr}_0".format(inst_nr=i) + endch
else:
    vadd_instantiation = ""

# Kernels Foorplaning
num_slr = 4 if args.board == "u250" else 2 if args.board == "u50" else 3
gt_slr = 1 if args.board == "u50" or args.board == "u55c" else 2
poe_slr = 3 if args.board == "u250" else gt_slr
cclo_slr = gt_slr if args.board == "u250" else gt_slr-1

if not args.axis3x and not args.hwemu:
    slr_constraints = "slr=cmac:SLR{slr_nr}\n".format(slr_nr=gt_slr)
else:
    slr_constraints = ""

for i in range(num_cclo):
    target_slr = min(i,num_slr-1) if args.axis3x else cclo_slr
    slr_constraints += "slr=arb_{inst_nr}:SLR{slr_nr}\nslr=arith_{inst_nr}:SLR{slr_nr}\nslr=ccl_offload_{inst_nr}:SLR{slr_nr}\n".format(inst_nr=i, slr_nr=target_slr)
    for j in range(3):
        slr_constraints += "slr=compression_{inst_nr}_{dp_nr}:SLR{slr_nr}\n".format(inst_nr=i, dp_nr=j, slr_nr=target_slr)
    for j in range(2):
        slr_constraints += "slr=hostctrl_{inst_nr}_{dp_nr}:SLR{slr_nr}\n".format(inst_nr=i, dp_nr=j, slr_nr=target_slr)
    if args.axis3x:
        slr_constraints += "slr=poe_{inst_nr}:SLR{slr_nr}\n".format(inst_nr=i, slr_nr=target_slr)
    else:
        slr_constraints += "slr=poe_0:SLR{slr_nr}\n".format(slr_nr=poe_slr)
    if not args.vadd:
        slr_constraints += "slr=lb_user_krnl_{inst_nr}:SLR{slr_nr}\n".format(inst_nr=i, slr_nr=target_slr)
    else:
        slr_constraints += "slr=vadd_{inst_nr}_0:SLR{slr_nr}\n".format(inst_nr=i, slr_nr=target_slr)
    for j in range(2):
        slr_constraints += "slr=extdma_{inst_nr}_{dp_nr}:SLR{slr_nr}\n".format(inst_nr=i, dp_nr=j, slr_nr=target_slr)

if args.poe == "tcp":
    slr_constraints += "slr=session_handler_0:SLR{slr_nr}\n".format(slr_nr=poe_slr)
    slr_constraints += "slr=lb_udp_txrx:SLR{slr_nr}\nslr=lb_udp_meta:SLR{slr_nr}\n".format(slr_nr=poe_slr)

# Memory bank assignment
mem_type = "DDR" if args.board == "u250" or args.board == "u200" else "HBM"
poe_ddr_bank = 3 if args.board == "u200" else poe_slr
cclo_ddr_bank = cclo_slr

mem_constraints = ""
bank_ctr = 0
for i in range(num_cclo):
    if mem_type == "DDR":
        target_bank = i if args.axis3x else cclo_slr
        mem_constraints += "sp=extdma_{inst_nr}_0.m_axi_0:DDR[{start_bank}]\n".format(inst_nr=i, start_bank=target_bank)
        mem_constraints += "sp=extdma_{inst_nr}_1.m_axi_0:DDR[{start_bank}]\n".format(inst_nr=i, start_bank=target_bank)
    else:
        mem_constraints += "sp=extdma_{inst_nr}_0.m_axi_0:HBM[{start_bank}:{end_bank}]\n".format(inst_nr=i, start_bank=bank_ctr, end_bank=bank_ctr+5)
        mem_constraints += "sp=extdma_{inst_nr}_1.m_axi_0:HBM[{start_bank}:{end_bank}]\n".format(inst_nr=i, start_bank=bank_ctr, end_bank=bank_ctr+5)
        bank_ctr += 6   

    if args.host:
        mem_constraints += "sp=extdma_{inst_nr}_0.m_axi_1:HOST[0]\n".format(inst_nr=i)
        mem_constraints += "sp=extdma_{inst_nr}_1.m_axi_1:HOST[0]\n".format(inst_nr=i)

    if args.poe == "tcp":
        poe_bank = bank_ctr if mem_type == "HBM" else poe_ddr_bank
        mem_constraints += "sp=poe_{inst_nr}.m00_axi:{mtype}[{start_bank}]\n".format(inst_nr=i, mtype=mem_type, start_bank=poe_bank)
        mem_constraints += "sp=poe_{inst_nr}.m01_axi:{mtype}[{start_bank}]\n".format(inst_nr=i, mtype=mem_type, start_bank=poe_bank)
        bank_ctr += 1 if mem_type == "HBM" else 0

# Stream connectivity
if args.axis3x:
    stream_connections = "stream_connect=poe_0.net_tx:poe_1.net_rx\n"
    stream_connections += "stream_connect=poe_1.net_tx:poe_2.net_rx\n"
    stream_connections += "stream_connect=poe_2.net_tx:poe_0.net_rx\n"
elif args.poe == "tcp":
    stream_connections = "stream_connect=poe_0.m_axis_udp_rx:lb_udp_txrx.in\n"
    stream_connections += "stream_connect=lb_udp_txrx.out:poe_0.s_axis_udp_tx\n"
    stream_connections += "stream_connect=poe_0.m_axis_udp_rx_meta:lb_udp_meta.in\n"
    stream_connections += "stream_connect=lb_udp_meta.out:poe_0.s_axis_udp_tx_meta\n"
    if args.hwemu:
        stream_connections += "stream_connect=gt_master.M00_AXIS:poe_0.axis_net_rx\n"
        stream_connections += "stream_connect=poe_0.axis_net_tx:gt_slave.S00_AXIS\n"
    else:
        stream_connections += "stream_connect=cmac.axis_net_rx:poe_0.axis_net_rx\n"
        stream_connections += "stream_connect=poe_0.axis_net_tx:cmac.axis_net_tx\n"
elif args.poe == "udp":
    if args.hwemu:
        stream_connections = "stream_connect=gt_master.M00_AXIS:poe_0.S_AXIS_eth2nl\n"
        stream_connections += "stream_connect=poe_0.M_AXIS_nl2eth:gt_slave.S00_AXIS\n"
    else:
        stream_connections = "stream_connect=cmac.M_AXIS:poe_0.S_AXIS_eth2nl\n"
        stream_connections += "stream_connect=poe_0.M_AXIS_nl2eth:cmac.S_AXIS\n"
else:
    if args.hwemu:
        stream_connections = "stream_connect=gt_master.M00_AXIS:poe_0.rx\n"
        stream_connections += "stream_connect=poe_0.tx:gt_slave.S00_AXIS\n"
    else:
        stream_connections = "stream_connect=cmac.M_AXIS:poe_0.rx\n"
        stream_connections += "stream_connect=poe_0.tx:cmac.S_AXIS\n"


# Connect host controllers to arbiter to CCL Offload, and connect plug-ins
for i in range(num_cclo):
    # Command interfaces
    stream_connections += "stream_connect=hostctrl_{inst_nr}_0.cmd:arb_{inst_nr}.cmd_clients_0\n".format(inst_nr=i)
    stream_connections += "stream_connect=hostctrl_{inst_nr}_1.cmd:arb_{inst_nr}.cmd_clients_1\n".format(inst_nr=i)
    stream_connections += "stream_connect=arb_{inst_nr}.cmd_cclo:ccl_offload_{inst_nr}.s_axis_call_req\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_call_ack:arb_{inst_nr}.ack_cclo\n".format(inst_nr=i)
    stream_connections += "stream_connect=arb_{inst_nr}.ack_clients_0:hostctrl_{inst_nr}_0.sts\n".format(inst_nr=i)
    stream_connections += "stream_connect=arb_{inst_nr}.ack_clients_1:hostctrl_{inst_nr}_1.sts\n".format(inst_nr=i)
    # Plugin interfaces
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_arith_op0:arith_{inst_nr}.in0\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_arith_op1:arith_{inst_nr}.in1\n".format(inst_nr=i)
    stream_connections += "stream_connect=arith_{inst_nr}.out_r:ccl_offload_{inst_nr}.s_axis_arith_res\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_compression0:compression_{inst_nr}_0.in_r\n".format(inst_nr=i)
    stream_connections += "stream_connect=compression_{inst_nr}_0.out_r:ccl_offload_{inst_nr}.s_axis_compression0\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_compression1:compression_{inst_nr}_1.in_r\n".format(inst_nr=i)
    stream_connections += "stream_connect=compression_{inst_nr}_1.out_r:ccl_offload_{inst_nr}.s_axis_compression1\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_compression2:compression_{inst_nr}_2.in_r\n".format(inst_nr=i)
    stream_connections += "stream_connect=compression_{inst_nr}_2.out_r:ccl_offload_{inst_nr}.s_axis_compression2\n".format(inst_nr=i)
    # Kernel interface
    if args.vadd:
        stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_krnl:vadd_{inst_nr}_0.data_from_cclo\n".format(inst_nr=i)
        stream_connections += "stream_connect=vadd_{inst_nr}_0.data_to_cclo:ccl_offload_{inst_nr}.s_axis_krnl\n".format(inst_nr=i)
    else:
        stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_krnl:lb_user_krnl_{inst_nr}.in\n".format(inst_nr=i)
        stream_connections += "stream_connect=lb_user_krnl_{inst_nr}.out:ccl_offload_{inst_nr}.s_axis_krnl\n".format(inst_nr=i)
    # External DMA interface
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma0_s2mm:extdma_{inst_nr}_0.s_axis_s2mm\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma0_mm2s_cmd:extdma_{inst_nr}_0.s_axis_mm2s_cmd\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma0_s2mm_cmd:extdma_{inst_nr}_0.s_axis_s2mm_cmd\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma1_s2mm:extdma_{inst_nr}_1.s_axis_s2mm\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma1_mm2s_cmd:extdma_{inst_nr}_1.s_axis_mm2s_cmd\n".format(inst_nr=i)
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_dma1_s2mm_cmd:extdma_{inst_nr}_1.s_axis_s2mm_cmd\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_0.m_axis_mm2s:ccl_offload_{inst_nr}.s_axis_dma0_mm2s\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_0.m_axis_mm2s_sts:ccl_offload_{inst_nr}.s_axis_dma0_mm2s_sts\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_0.m_axis_s2mm_sts:ccl_offload_{inst_nr}.s_axis_dma0_s2mm_sts\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_1.m_axis_mm2s:ccl_offload_{inst_nr}.s_axis_dma1_mm2s\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_1.m_axis_mm2s_sts:ccl_offload_{inst_nr}.s_axis_dma1_mm2s_sts\n".format(inst_nr=i)
    stream_connections += "stream_connect=extdma_{inst_nr}_1.m_axis_s2mm_sts:ccl_offload_{inst_nr}.s_axis_dma1_s2mm_sts\n".format(inst_nr=i)

# Connect CCLOs to POEs
if args.poe == "tcp" or args.poe == "tcp_dummy":
    for i in range(num_cclo):
        stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_notification:ccl_offload_{inst_nr}.s_axis_eth_notification:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_rx_meta:ccl_offload_{inst_nr}.s_axis_eth_rx_meta:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_rx_data:ccl_offload_{inst_nr}.s_axis_eth_rx_data:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_tx_status:ccl_offload_{inst_nr}.s_axis_eth_tx_status:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_eth_read_pkg:poe_{inst_nr}.s_axis_tcp_read_pkg:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_eth_tx_meta:poe_{inst_nr}.s_axis_tcp_tx_meta:512\n".format(inst_nr=i)
        stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_eth_tx_data:poe_{inst_nr}.s_axis_tcp_tx_data:512\n".format(inst_nr=i)
        if args.poe == "tcp":
            stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_port_status:session_handler_{inst_nr}.port_status:512\n".format(inst_nr=i)
            stream_connections += "stream_connect=poe_{inst_nr}.m_axis_tcp_open_status:session_handler_{inst_nr}.open_status:512\n".format(inst_nr=i)
            stream_connections += "stream_connect=session_handler_{inst_nr}.listen_port:poe_{inst_nr}.s_axis_tcp_listen_port:512\n".format(inst_nr=i)
            stream_connections += "stream_connect=session_handler_{inst_nr}.open_connection:poe_{inst_nr}.s_axis_tcp_open_connection:512\n".format(inst_nr=i)
            stream_connections += "stream_connect=session_handler_{inst_nr}.close_connection:poe_{inst_nr}.s_axis_tcp_close_connection:512\n".format(inst_nr=i)
elif args.poe == "udp":
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_eth_tx_data:poe_0.S_AXIS_sk2nl:512\n".format(inst_nr=i)
    stream_connections += "stream_connect=poe_0.M_AXIS_nl2sk:ccl_offload_{inst_nr}.s_axis_eth_rx_data:512\n".format(inst_nr=i)
else:
    stream_connections += "stream_connect=ccl_offload_{inst_nr}.m_axis_eth_tx_data:poe_0.inputData:512\n".format(inst_nr=i)
    stream_connections += "stream_connect=poe_0.outData:ccl_offload_{inst_nr}.s_axis_eth_rx_data:512\n".format(inst_nr=i)

with open(args.outfile, "w") as f:
    f.write("[connectivity]\n")
    f.write(cclo_instantiation+"\n")
    f.write(extdma_instantiation+"\n")
    f.write(arb_instantiation+"\n")
    f.write(hc_instantiation+"\n")
    f.write(reduce_instantiation+"\n")
    f.write(cast_instantiation+"\n")
    f.write(poe_instantiation+"\n")
    f.write(cmac_instantiation+"\n")
    f.write(loopback_instantiation+"\n")
    f.write(vadd_instantiation+"\n")
    if not args.hwemu:
        f.write(slr_constraints+"\n")
    f.write(mem_constraints+"\n")
    f.write(stream_connections+"\n")


