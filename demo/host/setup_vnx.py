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

import pynq
import numpy as np
from _thread import *
import threading 
import socket
from vnx_utils import *


def configure_vnx_ip(overlay, our_ip):
    print("Link interface 1 {}".format(ol.cmac_1.linkStatus()))
    print(ol.networklayer_0.updateIPAddress(our_ip, debug=True))

def configure_vnx_socket(overlay, their_rank, our_port, their_ip, their_port):
    # populate socket table with tuples of remote ip, remote port, local port 
    # up to 16 entries possible in VNx
    ol.networklayer_0.sockets[their_rank] = (their_ip, their_port, our_port, True)
    ol.networklayer_0.populateSocketTable(debug=True)

def configure_vnx(overlay, localrank, ranks):
    assert len(ranks) <= 16, "Too many ranks. VNX supports up to 16 sockets"
    for i in range(len(ranks)):
        if i == localrank:
            configure_vnx_ip(overlay, ranks[i]["ip"])
        else:
            configure_vnx_socket(overlay, i, ranks[localrank]["port"], ranks[i]["ip"], ranks[i]["port"], True)
            
            