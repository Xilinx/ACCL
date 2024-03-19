# /*******************************************************************************
#  Copyright (C) 2024 Advanced Micro Devices, Inc
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
parser.add_argument('-n', '--numdma', type=int, default=1, help="Number of generated DMA interfaces")
args = parser.parse_args()

verilog_wrapper= """

`timescale 1 ns / 1 ps

module external_dma
(
  input ap_clk,
  input ap_rst_n,

  input [15:0] s_axi_control_araddr,
  output s_axi_control_arready,
  input s_axi_control_arvalid,
  input [15:0] s_axi_control_awaddr,
  output s_axi_control_awready,
  input s_axi_control_awvalid,
  input s_axi_control_bready,
  output [1:0] s_axi_control_bresp,
  output s_axi_control_bvalid,
  output [31:0] s_axi_control_rdata,
  input s_axi_control_rready,
  output [1:0] s_axi_control_rresp,
  output s_axi_control_rvalid,
  input [31:0] s_axi_control_wdata,
  output s_axi_control_wready,
  input [3:0] s_axi_control_wstrb,
  input s_axi_control_wvalid,

  {}

  input [511:0] s_axis_s2mm_tdata,
  input [63:0] s_axis_s2mm_tkeep,
  input [7:0] s_axis_s2mm_tdest,
  input s_axis_s2mm_tlast,
  output s_axis_s2mm_tready,
  input s_axis_s2mm_tvalid,

  output [511:0] m_axis_mm2s_tdata,
  output [63:0] m_axis_mm2s_tkeep,
  output m_axis_mm2s_tlast,
  input m_axis_mm2s_tready,
  output m_axis_mm2s_tvalid,

  input [103:0] s_axis_mm2s_cmd_tdata,
  output s_axis_mm2s_cmd_tready,
  input s_axis_mm2s_cmd_tvalid,
  input [7:0] s_axis_mm2s_cmd_tdest,

  output [7:0] m_axis_mm2s_sts_tdata,
  input m_axis_mm2s_sts_tready,
  output m_axis_mm2s_sts_tvalid,
  output [0:0] m_axis_mm2s_sts_tkeep,
  output m_axis_mm2s_sts_tlast,

  input [103:0] s_axis_s2mm_cmd_tdata,
  output s_axis_s2mm_cmd_tready,
  input s_axis_s2mm_cmd_tvalid,
  input [7:0] s_axis_s2mm_cmd_tdest,

  output [31:0] m_axis_s2mm_sts_tdata,
  input m_axis_s2mm_sts_tready,
  output m_axis_s2mm_sts_tvalid,
  output [3:0] m_axis_s2mm_sts_tkeep,
  output m_axis_s2mm_sts_tlast
);

  external_dma_bd ext_dma_bd(

        .s_axi_control_araddr(s_axi_control_araddr),
        .s_axi_control_arready(s_axi_control_arready),
        .s_axi_control_arvalid(s_axi_control_arvalid),
        .s_axi_control_awaddr(s_axi_control_awaddr),
        .s_axi_control_awready(s_axi_control_awready),
        .s_axi_control_awvalid(s_axi_control_awvalid),
        .s_axi_control_bready(s_axi_control_bready),
        .s_axi_control_bresp(s_axi_control_bresp),
        .s_axi_control_bvalid(s_axi_control_bvalid),
        .s_axi_control_rdata(s_axi_control_rdata),
        .s_axi_control_rready(s_axi_control_rready),
        .s_axi_control_rresp(s_axi_control_rresp),
        .s_axi_control_rvalid(s_axi_control_rvalid),
        .s_axi_control_wdata(s_axi_control_wdata),
        .s_axi_control_wready(s_axi_control_wready),
        .s_axi_control_wstrb(s_axi_control_wstrb),
        .s_axi_control_wvalid(s_axi_control_wvalid),

        {}

        .s_axis_s2mm_tdata(s_axis_s2mm_tdata),
        .s_axis_s2mm_tkeep(s_axis_s2mm_tkeep),
        .s_axis_s2mm_tdest(s_axis_s2mm_tdest),
        .s_axis_s2mm_tlast(s_axis_s2mm_tlast),
        .s_axis_s2mm_tready(s_axis_s2mm_tready),
        .s_axis_s2mm_tvalid(s_axis_s2mm_tvalid),

        .m_axis_mm2s_tdata(m_axis_mm2s_tdata),
        .m_axis_mm2s_tkeep(m_axis_mm2s_tkeep),
        .m_axis_mm2s_tlast(m_axis_mm2s_tlast),
        .m_axis_mm2s_tready(m_axis_mm2s_tready),
        .m_axis_mm2s_tvalid(m_axis_mm2s_tvalid),

        .s_axis_mm2s_cmd_tdata(s_axis_mm2s_cmd_tdata),
        .s_axis_mm2s_cmd_tready(s_axis_mm2s_cmd_tready),
        .s_axis_mm2s_cmd_tvalid(s_axis_mm2s_cmd_tvalid),
        .s_axis_mm2s_cmd_tdest(s_axis_mm2s_cmd_tdest),

        .m_axis_mm2s_sts_tdata(m_axis_mm2s_sts_tdata),
        .m_axis_mm2s_sts_tready(m_axis_mm2s_sts_tready),
        .m_axis_mm2s_sts_tvalid(m_axis_mm2s_sts_tvalid),
        .m_axis_mm2s_sts_tkeep(m_axis_mm2s_sts_tkeep),
        .m_axis_mm2s_sts_tlast(m_axis_mm2s_sts_tlast),

        .s_axis_s2mm_cmd_tdata(s_axis_s2mm_cmd_tdata),
        .s_axis_s2mm_cmd_tready(s_axis_s2mm_cmd_tready),
        .s_axis_s2mm_cmd_tvalid(s_axis_s2mm_cmd_tvalid),
        .s_axis_s2mm_cmd_tdest(s_axis_s2mm_cmd_tdest),

        .m_axis_s2mm_sts_tdata(m_axis_s2mm_sts_tdata),
        .m_axis_s2mm_sts_tready(m_axis_s2mm_sts_tready),
        .m_axis_s2mm_sts_tvalid(m_axis_s2mm_sts_tvalid),
        .m_axis_s2mm_sts_tkeep(m_axis_s2mm_sts_tkeep),
        .m_axis_s2mm_sts_tlast(m_axis_s2mm_sts_tlast),

        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n)
        );

endmodule
"""

axi_intf_declaration = """
  output [63:0] m_axi_{0}_araddr,
  output [1:0] m_axi_{0}_arburst,
  output [3:0] m_axi_{0}_arcache,
  output [7:0] m_axi_{0}_arlen,
  output [2:0] m_axi_{0}_arprot,
  input m_axi_{0}_arready,
  output [2:0] m_axi_{0}_arsize,
  output [3:0] m_axi_{0}_aruser,
  output m_axi_{0}_arvalid,
  output [63:0] m_axi_{0}_awaddr,
  output [1:0] m_axi_{0}_awburst,
  output [3:0] m_axi_{0}_awcache,
  output [7:0] m_axi_{0}_awlen,
  output [2:0] m_axi_{0}_awprot,
  input m_axi_{0}_awready,
  output [2:0] m_axi_{0}_awsize,
  output [3:0] m_axi_{0}_awuser,
  output m_axi_{0}_awvalid,
  output m_axi_{0}_bready,
  input [1:0] m_axi_{0}_bresp,
  input m_axi_{0}_bvalid,
  input [511:0] m_axi_{0}_rdata,
  input m_axi_{0}_rlast,
  output m_axi_{0}_rready,
  input [1:0] m_axi_{0}_rresp,
  input m_axi_{0}_rvalid,
  output [511:0] m_axi_{0}_wdata,
  output m_axi_{0}_wlast,
  input m_axi_{0}_wready,
  output [63:0] m_axi_{0}_wstrb,
  output m_axi_{0}_wvalid,

"""

axi_intf_connection = """
        .m_axi_{0}_araddr(m_axi_{0}_araddr),
        .m_axi_{0}_arburst(m_axi_{0}_arburst),
        .m_axi_{0}_arcache(m_axi_{0}_arcache),
        .m_axi_{0}_arlen(m_axi_{0}_arlen),
        .m_axi_{0}_arprot(m_axi_{0}_arprot),
        .m_axi_{0}_arready(m_axi_{0}_arready),
        .m_axi_{0}_arsize(m_axi_{0}_arsize),
        .m_axi_{0}_aruser(m_axi_{0}_aruser),
        .m_axi_{0}_arvalid(m_axi_{0}_arvalid),
        .m_axi_{0}_awaddr(m_axi_{0}_awaddr),
        .m_axi_{0}_awburst(m_axi_{0}_awburst),
        .m_axi_{0}_awcache(m_axi_{0}_awcache),
        .m_axi_{0}_awlen(m_axi_{0}_awlen),
        .m_axi_{0}_awprot(m_axi_{0}_awprot),
        .m_axi_{0}_awready(m_axi_{0}_awready),
        .m_axi_{0}_awsize(m_axi_{0}_awsize),
        .m_axi_{0}_awuser(m_axi_{0}_awuser),
        .m_axi_{0}_awvalid(m_axi_{0}_awvalid),
        .m_axi_{0}_bready(m_axi_{0}_bready),
        .m_axi_{0}_bresp(m_axi_{0}_bresp),
        .m_axi_{0}_bvalid(m_axi_{0}_bvalid),
        .m_axi_{0}_rdata(m_axi_{0}_rdata),
        .m_axi_{0}_rlast(m_axi_{0}_rlast),
        .m_axi_{0}_rready(m_axi_{0}_rready),
        .m_axi_{0}_rresp(m_axi_{0}_rresp),
        .m_axi_{0}_rvalid(m_axi_{0}_rvalid),
        .m_axi_{0}_wdata(m_axi_{0}_wdata),
        .m_axi_{0}_wlast(m_axi_{0}_wlast),
        .m_axi_{0}_wready(m_axi_{0}_wready),
        .m_axi_{0}_wstrb(m_axi_{0}_wstrb),
        .m_axi_{0}_wvalid(m_axi_{0}_wvalid),

"""

all_axi_declarations = ""
all_axi_connections = ""
for i in range(args.numdma):
    all_axi_declarations += axi_intf_declaration.format(i)
    all_axi_connections += axi_intf_connection.format(i)

with open("external_dma.v", "w") as f:
    f.write(verilog_wrapper.format(all_axi_declarations, all_axi_connections))

kernel_xml = """
<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="9">
<kernel name="external_dma" language="ip" vlnv="Xilinx:ACCL:external_dma:1.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" interrupt="false" hwControlProtocol="user_managed">
<ports>
<port name="s_axi_control" mode="slave" range="0x10000" dataWidth="32" portType="addressable" base="0x0"/>
<port name="s_axis_s2mm" mode="read_only" dataWidth="512" portType="stream"/>
<port name="m_axis_mm2s" mode="write_only" dataWidth="512" portType="stream"/>
<port name="s_axis_s2mm_cmd" mode="read_only" dataWidth="104" portType="stream"/>
<port name="s_axis_mm2s_cmd" mode="read_only" dataWidth="104" portType="stream"/>
<port name="m_axis_s2mm_sts" mode="write_only" dataWidth="32" portType="stream"/>
<port name="m_axis_mm2s_sts" mode="write_only" dataWidth="8" portType="stream"/>
{}</ports>
<args>
<arg name="s_axis_s2mm" addressQualifier="4" id="0" port="s_axis_s2mm" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
<arg name="m_axis_mm2s" addressQualifier="4" id="1" port="m_axis_mm2s" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
<arg name="s_axis_s2mm_cmd" addressQualifier="4" id="2" port="s_axis_s2mm_cmd" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
<arg name="s_axis_mm2s_cmd" addressQualifier="4" id="3" port="s_axis_mm2s_cmd" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
<arg name="m_axis_s2mm_sts" addressQualifier="4" id="4" port="m_axis_s2mm_sts" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
<arg name="m_axis_mm2s_sts" addressQualifier="4" id="5" port="m_axis_mm2s_sts" size="0x4" offset="0x0" hostOffset="0x0" hostSize="0x4" type="void*" />
{}</args>
</kernel></root>
"""

xml_axi_port = """<port name="m_axi_{0}" mode="master" range="0xFFFFFFFFFFFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>
"""

xml_axi_arg = """<arg name="m_axi_{0}" addressQualifier="1" id="{1}" port="m_axi_{0}" size="0x8" offset="0x20" hostOffset="0x0" hostSize="0x8" type="int*" />
"""

all_xml_ports = ""
all_xml_args = ""
for i in range(args.numdma):
    all_xml_ports += xml_axi_port.format(i)
    all_xml_args += xml_axi_arg.format(i,i+6)

with open("kernel.xml", "w") as f:
    f.write(kernel_xml.format(all_xml_ports, all_xml_args))
