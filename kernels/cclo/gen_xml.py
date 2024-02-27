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

import sys

def fill_xml_stream_port_arg(ports, args, name, is_master, width, id):
    port_spec = ports + "<port name=\"%s\" mode=\"%s\" dataWidth=\"%d\" portType=\"stream\"/>\n" % (name, "write_only" if is_master else "read_only", width)
    arg_spec = args + "<arg name=\"%s\" addressQualifier=\"4\" id=\"%d\" port=\"%s\" size=\"0x4\" offset=\"0x0\" hostOffset=\"0x0\" hostSize=\"0x4\" type=\"void*\" />\n" % (name, id, name)
    return port_spec, arg_spec, id + 1

def fill_xml_axilite_port(name, range):
    return "<port name=\"%s\" mode=\"slave\" range=\"%s\" dataWidth=\"32\" portType=\"addressable\" base=\"0x0\"/>\n" % (name, hex(range))

def fill_xml_aximm_port_arg(ports, args, name, width, offset, id):
    port_spec = ports + "<port name=\"%s\" mode=\"master\" range=\"0xFFFFFFFFFFFFFFFF\" dataWidth=\"%d\" portType=\"addressable\" base=\"0x0\"/>\n" % (name, width)
    arg_spec = args + "<arg name=\"%s\" addressQualifier=\"1\" id=\"%d\" port=\"%s\" size=\"0x8\" offset=\"%s\" hostOffset=\"0x0\" hostSize=\"0x8\" type=\"int*\" />\n" % (name, id, name, hex(offset))
    return port_spec, arg_spec, id + 1

xml_header = """<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="9">
<kernel name="ccl_offload" language="ip" vlnv="Xilinx:ACCL:ccl_offload:1.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" interrupt="false" hwControlProtocol="user_managed">
"""
xml_footer = "</kernel></root>"

xml_ports = "<ports>\n"
xml_args = "<args>\n"

#$(STACK_TYPE) $(EN_DMA) $(EN_ARITH) $(EN_COMPRESS) $(EN_EXT_KRNL)

xml_ports += fill_xml_axilite_port("s_axi_control", 8*1024)

id = 0

xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_call_req", False, 32, id)
xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_call_ack", True, 32, id)
xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_eth_rx_data", False, 512, id)
xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_eth_tx_data", True, 512, id)

if sys.argv[1] == "TCP":
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_eth_notification", False, 128, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_eth_read_pkg", True, 32, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_eth_rx_meta", False, 16, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_eth_tx_meta", True, 32, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_eth_tx_status", False, 64, id)

if sys.argv[1] == "RDMA":
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_eth_notification", False, 64, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_rdma_sq", True, 128, id)

if  int(sys.argv[5]) == 1:
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_krnl", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_krnl", True, 512, id)

if  int(sys.argv[4]) == 1:
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_compression0", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_compression0", True, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_compression1", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_compression1", True, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_compression2", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_compression2", True, 512, id)

if  int(sys.argv[3]) == 1:
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_arith_res", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_arith_op0", True, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_arith_op1", True, 512, id)

if  int(sys.argv[2]) == 1:
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma0_s2mm", True, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma0_mm2s", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma1_s2mm", True, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma1_mm2s", False, 512, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma0_mm2s_cmd", True, 104, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma0_mm2s_sts", False, 32, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma0_s2mm_cmd", True, 104, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma0_s2mm_sts", False, 32, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma1_mm2s_cmd", True, 104, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma1_mm2s_sts", False, 32, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "m_axis_dma1_s2mm_cmd", True, 104, id)
    xml_ports, xml_args, id = fill_xml_stream_port_arg(xml_ports, xml_args, "s_axis_dma1_s2mm_sts", False, 32, id)

xml_ports += "</ports>\n"
xml_args += "</args>\n"

with open("ccl_offload.xml", "w") as f:
    f.write(xml_header + xml_ports + xml_args + xml_footer)
