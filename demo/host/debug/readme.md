# Reconstruct CMAC frame
## Install requirements

``pip install -r requirements.txt``

## Inspect frames going through the cmac
1. Attach an ILA to the cmac
2. Collect data through an ILA attached to the cmac.
3. Export the ILA data as vcd
4. run ``cmac_inspector.py`` 

Arguments:
- You need to provide the vcd file as first argument to the python script.
- optional: cmac_rx/cmac_tx these to parameters specify the ILA slot that are connected to the cmac. For the exact name you should take a look at vcd preamble.For example:

.vcd
````
$date
        2021-Sep-22 17:28:25
$end
$version
        Vivado v2020.2 (64-bit)
$end
$timescale
        1ps
$end
$scope module dut $end
$var reg 512 " pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0_axis_tdata [511:0] $end
$var reg 64 L& pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0_axis_tkeep [63:0] $end
$var reg 1 .' pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0_axis_tvalid $end
$var reg 1 /' pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0_axis_tready $end
$var reg 1 0' pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0_axis_tlast $end
$var reg 512 1' pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1_axis_tdata [511:0] $end
$var reg 64 [, pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1_axis_tkeep [63:0] $end
$var reg 1 =- pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1_axis_tvalid $end
$var reg 1 >- pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1_axis_tready $end
$var reg 1 ?- pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1_axis_tlast $end
$var reg 128 @- pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_2_axis_tdata [127:0] $end
$var reg 16 b. pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_2_axis_tkeep [15:0] $end
$var reg 1 r. pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_2_axis_tvalid $end
$var reg 1 s. pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_2_axis_tready $end
$var reg 1 t. pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_2_axis_tlast $end
$var reg 128 u. pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_3_axis_tdata [127:0] $end
$var reg 16 90 pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_3_axis_tkeep [15:0] $end
$var reg 1 I0 pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_3_axis_tvalid $end
$var reg 1 J0 pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_3_axis_tready $end
...
````
And in our case the cmac_rx/tx is connected to slot0 /slot1 under dut scope so you would use 
````
--cmac_rx dut.pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0
--cmac_tx dut.pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1
````
- optional: ext switch to request frame parsing through an external tool 




Example 
````
$ python cmac_inspector.py network_stack.vcd
finish tx
rx 1  FRAME: dst addr: 33:33:00:10:00:20 src addr: 09:2e:ab:ad:41:43 type: 86dd rest: 60000000006d1101fe800000000000004c7d71f540d6a04fff02000000000000000000000001000202220223006dbfa1014941720008000200650001000e000100012087c95990e2bada14340003000c1290e2ba000000000000000000270017000a6361746170756c7430320164046574687a026368000010000e0000013700084d53465420352e3000060008001800170011002728a80c00001e000000000000001e000000000000001e00000000000000
````

Example with external online frame reconstruction tool based on wireshark

````
g$ python cmac_inspector.py network_stack.vcd --ext                                                                                                  finish tx                                                                                                                                                                                                                                    rx 1  Frame 1: 192 bytes on wire (1536 bits), 192 bytes captured (1536 bits)                                                                                                                                                                     Encapsulation type: Ethernet (1)                                                                                                                                                                                                             Arrival Time: Sep 29, 2021 13:25:35.000000000 CEST                                                                                                                                                                                           [Time shift for this packet: 0.000000000 seconds]                                                                                                                                                                                            Epoch Time: 1632914735.000000000 seconds                                                                                                                                                                                                     [Time delta from previous captured frame: 0.000000000 seconds]                                                                                                                                                                               [Time delta from previous displayed frame: 0.000000000 seconds]                                                                                                                                                                              [Time since reference or first frame: 0.000000000 seconds]                                                                                                                                                                                   Frame Number: 1                                                                                                                                                                                                                              Frame Length: 192 bytes (1536 bits)                                                                                                                                                                                                          Capture Length: 192 bytes (1536 bits)                                                                                                                                                                                                        [Frame is marked: False]                                                                                                                                                                                                                     [Frame is ignored: False]                                                                                                                                                                                                                    [Protocols in frame: eth:ethertype:ipv6:udp:dhcpv6]                                                                                                                                                                                      Ethernet II, Src: IntelCor_da:14:34 (90:e2:ba:da:14:34), Dst: IPv6mcast_01:00:02 (33:33:00:01:00:02)                                                                                                                                             Destination: IPv6mcast_01:00:02 (33:33:00:01:00:02)                                                                                                                                                                                              Address: IPv6mcast_01:00:02 (33:33:00:01:00:02)                                                                                                                                                                                              .... ..1. .... .... .... .... = LG bit: Locally administered address (this is NOT the factory default)                                                                                                                                       .... ...1 .... .... .... .... = IG bit: Group address (multicast/broadcast)                                                                                                                                                              Source: IntelCor_da:14:34 (90:e2:ba:da:14:34)                                                                                                                                                                                                    Address: IntelCor_da:14:34 (90:e2:ba:da:14:34)                                                                                                                                                                                               .... ..0. .... .... .... .... = LG bit: Globally unique address (factory default)                                                                                                                                                            .... ...0 .... .... .... .... = IG bit: Individual address (unicast)                                                                                                                                                                     Type: IPv6 (0x86dd)                                                                                                                                                                                                                          Trailer: 28a80c00001e000000000000001e000000000000001e000000                                                                                                                                                                                  Frame check sequence: 0x00000000 [unverified]                                                                                                                                                                                                [FCS Status: Unverified]
Internet Protocol Version 6, Src: fe80::4c7d:71f5:40d6:a04f, Dst: ff02::1:2                                                                                                                                                                      0110 .... = Version: 6                                                                                                                                                                                                                       .... 0000 0000 .... .... .... .... .... = Traffic Class: 0x00 (DSCP: CS0, ECN: Not-ECT)                                                                                                                                                          .... 0000 00.. .... .... .... .... .... = Differentiated Services Codepoint: Default (0)                                                                                                                                                     .... .... ..00 .... .... .... .... .... = Explicit Congestion Notification: Not ECN-Capable Transport (0)                                                                                                                                .... .... .... 0000 0000 0000 0000 0000 = Flow Label: 0x00000                                                                                                                                                                                Payload Length: 109                                                                                                                                                                                                                          Next Header: UDP (17)                                                                                                                                                                                                                        Hop Limit: 1                                                                                                                                                                                                                                 Source Address: fe80::4c7d:71f5:40d6:a04f                                                                                                                                                                                                    Destination Address: ff02::1:2
User Datagram Protocol, Src Port: 546, Dst Port: 547                                                                                                                                                                                             Source Port: 546                                                                                                                                                                                                                             Destination Port: 547                                                                                                                                                                                                                        Length: 109                                                                                                                                                                                                                                  Checksum: 0xbfa1 [unverified]                                                                                                                                                                                                                [Checksum Status: Unverified]                                                                                                                                                                                                                [Stream index: 0]                                                                                                                                                                                                                            [Timestamps]
        [Time since first frame: 0.000000000 seconds]
````