
def reverse_str(s):
    return s[::-1]
#reorder
def reorder(s):
    res = ""
    for i in range(0,len(s), 128):
        an_array = bytearray.fromhex(s[i:i+128])
        an_array.reverse()

        res += "".join(format(x, '02x') for x in an_array) 
    return res

ENDLN = " "
    
def print_frame(s):
	if args.raw:
		print(s)
	if args.ext:
		import requests
		#print(f"https://hpd.gasmi.net/api.php?format=text&data{s}")
		r = requests.get('https://hpd.gasmi.net/api.php', params={"format":"text", "data":s})
		print(r.text)

		return
	print("FRAME:", end=ENDLN)
	print("dst addr:", end=ENDLN); print_MAC(s[0 :12])
	print("src addr:", end=ENDLN); print_MAC(s[12:24])
	frame_type = s[24:28]
	if frame_type == "0800":
		print("type: IPV4", end=ENDLN)
		print_datagram(s[28:])
	elif frame_type == "0806" :
		print("type: ARP", end=ENDLN)
		print_ARP(s[28:])
	else:
		print("type:",frame_type , end=ENDLN)
		print("rest:", s[28:])

def print_MAC(s):
    
    for i in range(5):
        print(reverse_str(s[i*2:i*2+2]),":", end="", sep="")
    print(reverse_str(s[10:]), end=ENDLN)


def print_ARP(s):
    hw          = s[0:4]
    print("HW:", hw, end=ENDLN)
    proto       = s[4:8]
    if proto == "0800":
        print("proto: IPV4", end=ENDLN)
    else:
        print("proto:", s[4:8], end=ENDLN)
    hw_addr_len     = int(s[8:10],16)
    proto_addr_len  = int(s[10:12],16)
    op_code     = s[12:16]
    print("hw   addr len:", hw_addr_len, end=ENDLN)
    print("prot addr len:", proto_addr_len, end=ENDLN)
    print(print_ARP_OPCODE(op_code))
    src_addr    = s[16:16+hw_addr_len]
    src_proto   = s[16+hw_addr_len:16+hw_addr_len+proto_addr_len]
    dst_addr    = s[16+hw_addr_len+proto_addr_len:16+hw_addr_len+proto_addr_len+hw_addr_len]
    dst_proto   = s[16+hw_addr_len+proto_addr_len+hw_addr_len:16+hw_addr_len+proto_addr_len+hw_addr_len+proto_addr_len]
    import ipaddress
    print("src hw:"     , print_MAC(src_addr) , end=ENDLN)
    print("src proto:"  , ipaddress.IPv4Address(int(src_proto,16)) if proto == "0800" else dst_proto, end=ENDLN)
    print("dst hw:"     , print_MAC(dst_addr) , end=ENDLN)
    print("dst proto:"  , ipaddress.IPv4Address(int(dst_proto,16)) if proto == "0800" else dst_proto, end=ENDLN)

def print_ARP_OPCODE(s):
    dictionary = {   
        1	:"ARP Request",
    	2	:"ARP Reply",
    	3	:"RARP Request",
    	4	:"RARP Reply",
    	5	:"DRARP Request",
    	6	:"DRARP Reply",
    	7	:"DRARP Error",
    	8	:"InARP Request",
    	9	:"InARP Reply",

    }
    code = int(s,16)

    if code <= 0 or code >= len(dictionary.keys()):
        return f"UNKNOWN ({code})"
    return int(s,16),dictionary[code]

def print_datagram(s):
    print("DATAGRAM:", end=ENDLN)
    
    fragment_offset = int(reverse_bytes(s[13:16]), 16)
    print("Version", s[0], "IHL", s[1], end=ENDLN)
    print("Type of service", s[2:4], end=ENDLN)
    print("Len:", s[4:8] , end=ENDLN)
    print("ID",   s[8:12], end=ENDLN)
    print("FLAGs:", s[12], end=ENDLN)
    print("Fragment offset:", fragment_offset, end=ENDLN)
    print("TTL", s[16:18], end=ENDLN)
    protocol = (s[18:20])
    print("Protocol", protocol_to_str(protocol), end=ENDLN)
    print("Header checksum", (s[20:24]), end=ENDLN)
    print("Src addr", s_to_ip( s[24:32]), end=ENDLN)
    print("Dst addr", s_to_ip( s[32:40]), end=ENDLN)
    if fragment_offset != 0:
        print("Options+padding" , s[40:40+fragment_offset], end=ENDLN)

    if protocol == "06":
        print_TCP(s[40+fragment_offset:])
    elif protocol == "11":
        print_UDP(s[40+fragment_offset:])
    else:   
        print(s[40+fragment_offset:], end=ENDLN)

     

def reverse_bytes(s):
    l = []
    for i in range(0, len(s), 2):
        if i +1  < len(s):
            l.append(s[i+1])
        l.append(s[i])
    return "".join(l)

def s_to_ip(s):
    
    import ipaddress
    return ipaddress.IPv4Address(int(s, 16))

def protocol_to_str(s):
    #https://en.wikipedia.org/wiki/List_of_IP_protocol_numbers
    dictionary = {   0	:"0	HOPOPT	IPv6 Hop-by-Hop Option	RFC 8200",
    	1	:"ICMP	Internet Control Message Protocol	RFC 792",
    	2	:"IGMP	Internet Group Management Protocol	RFC 1112",
    	3	:"GGP	Gateway-to-Gateway Protocol	RFC 823",
    	4	:"IP-in-IP	IP in IP (encapsulation)	RFC 2003",
    	5	:"ST	Internet Stream Protocol	RFC 1190, RFC 1819",
    	6	:"TCP	Transmission Control Protocol	RFC 793",
    	7	:"CBT	Core-based trees	RFC 2189",
    	8	:"EGP	Exterior Gateway Protocol	RFC 888",
    	9	:"IGP	Interior Gateway Protocol (any private interior gateway, for example Cisco's IGRP)	",
    	10	:"BBN-RCC-MON	BBN RCC Monitoring	",
    	11	:"NVP-II	Network Voice Protocol	RFC 741",
    	12	:"PUP	Xerox PUP	",
    	13	:"ARGUS	ARGUS	",
    	14	:"EMCON	EMCON	",
    	15	:"XNET	Cross Net Debugger	IEN 158[2]",
    	16	:"CHAOS	Chaos	",
    	17	:"UDP	User Datagram Protocol	RFC 768",
    	18	:"MUX	Multiplexing	IEN 90[3]",
    	19	:"DCN-MEAS	DCN Measurement Subsystems	",
    	20	:"HMP	Host Monitoring Protocol	RFC 869",
    	21	:"PRM	Packet Radio Measurement	",
    	22	:"XNS-IDP	XEROX NS IDP	",
    	23	:"TRUNK-1	Trunk-1	",
    	24	:"TRUNK-2	Trunk-2	",
    	25	:"LEAF-1	Leaf-1	",
    	26	:"LEAF-2	Leaf-2	",
    	27	:"RDP	Reliable Data Protocol	RFC 908",
    	28	:"IRTP	Internet Reliable Transaction Protocol	RFC 938",
    	29	:"ISO-TP4	ISO Transport Protocol Class 4	RFC 905",
    	30	:"NETBLT	Bulk Data Transfer Protocol	RFC 998",
    	31	:"MFE-NSP	MFE Network Services Protocol	",
    	32	:"MERIT-INP	MERIT Internodal Protocol	",
    	33	:"DCCP	Datagram Congestion Control Protocol	RFC 4340",
    	34	:"3PC	Third Party Connect Protocol	",
    	35	:"IDPR	Inter-Domain Policy Routing Protocol	RFC 1479",
    	36	:"XTP	Xpress Transport Protocol	",
    	37	:"DDP	Datagram Delivery Protocol	",
    	38	:"IDPR-CMTP	IDPR Control Message Transport Protocol	",
    	39	:"TP++	TP++ Transport Protocol	",
    	40	:"IL	IL Transport Protocol	",
    	41	:"IPv6	IPv6 Encapsulation	RFC 2473",
    	42	:"SDRP	Source Demand Routing Protocol	RFC 1940",
    	43	:"IPv6-Route	Routing Header for IPv6	RFC 8200",
    	44	:"IPv6-Frag	Fragment Header for IPv6	RFC 8200",
    	45	:"IDRP	Inter-Domain Routing Protocol	",
    	46	:"RSVP	Resource Reservation Protocol	RFC 2205",
    	47	:"GRE	Generic Routing Encapsulation	RFC 2784, RFC 2890",
    	48	:"DSR	Dynamic Source Routing Protocol	RFC 4728",
    	49	:"BNA	Burroughs Network Architecture	",
    	50	:"ESP	Encapsulating Security Payload	RFC 4303",
    	51	:"AH	Authentication Header	RFC 4302",
    	52	:"I-NLSP	Integrated Net Layer Security Protocol	TUBA",
    	53	:"SwIPe	SwIPe	RFC 5237",
    	54	:"NARP	NBMA Address Resolution Protocol	RFC 1735",
    	55	:"MOBILE	IP Mobility (Min Encap)	RFC 2004",
    	56	:"TLSP	Transport Layer Security Protocol (using Kryptonet key management)	",
    	57	:"SKIP	Simple Key-Management for Internet Protocol	RFC 2356",
    	58	:"IPv6-ICMP	ICMP for IPv6	RFC 4443, RFC 4884",
    	59	:"IPv6-NoNxt	No Next Header for IPv6	RFC 8200",
    	60	:"IPv6-Opts	Destination Options for IPv6	RFC 8200",
    	61	:"	Any host internal protocol	",
    	62	:"CFTP	CFTP	",
    	63	:"	Any local network	",
    	64	:"SAT-EXPAK	SATNET and Backroom EXPAK	",
    	65	:"KRYPTOLAN	Kryptolan	",
    	66	:"RVD	MIT Remote Virtual Disk Protocol	",
    	67	:"IPPC	Internet Pluribus Packet Core	",
    	68	:"	Any distributed file system	",
    	69	:"SAT-MON	SATNET Monitoring	",
    	70	:"VISA	VISA Protocol	",
    	71	:"IPCU	Internet Packet Core Utility	",
    	72	:"CPNX	Computer Protocol Network Executive	",
    	73	:"CPHB	Computer Protocol Heart Beat	",
    	74	:"WSN	Wang Span Network	",
    	75	:"PVP	Packet Video Protocol	",
    	76	:"BR-SAT-MON	Backroom SATNET Monitoring	",
    	77	:"SUN-ND	SUN ND PROTOCOL-Temporary	",
    	78	:"WB-MON	WIDEBAND Monitoring	",
    	79	:"WB-EXPAK	WIDEBAND EXPAK	",
    	80	:"ISO-IP	International Organization for Standardization Internet Protocol	",
    	81	:"VMTP	Versatile Message Transaction Protocol	RFC 1045",
    	82	:"SECURE-VMTP	Secure Versatile Message Transaction Protocol	RFC 1045",
    	83	:"VINES	VINES	",
    	84	:"TTP	TTP	",
    	84	:"IPTM	Internet Protocol Traffic Manager	",
    	85	:"NSFNET-IGP	NSFNET-IGP	",
    	86	:"DGP	Dissimilar Gateway Protocol	",
    	87	:"TCF	TCF	",
    	88	:"EIGRP	EIGRP	Informational RFC 7868",
    	89	:"OSPF	Open Shortest Path First	RFC 2328",
    	90	:"Sprite-RPC	Sprite RPC Protocol	",
    	91	:"LARP	Locus Address Resolution Protocol	",
    	92	:"MTP	Multicast Transport Protocol	",
    	93	:"AX.25	AX.25	",
    	94	:"OS	KA9Q NOS compatible IP over IP tunneling	",
    	95	:"MICP	Mobile Internetworking Control Protocol	",
    	96	:"SCC-SP	Semaphore Communications Sec. Pro	",
    	97	:"ETHERIP	Ethernet-within-IP Encapsulation	RFC 3378",
    	98	:"ENCAP	Encapsulation Header	RFC 1241",
    	99	:"	Any private encryption scheme	",
    	100	:"GMTP	GMTP	",
    	101	:"IFMP	Ipsilon Flow Management Protocol	",
    	102	:"PNNI	PNNI over IP	",
    	103	:"PIM	Protocol Independent Multicast	",
    	104	:"ARIS	IBM's ARIS (Aggregate Route IP Switching) Protocol	",
    	105	:"SCPS	SCPS (Space Communications Protocol Standards)	SCPS-TP[4]",
    	106	:"QNX	QNX	",
    	107	:"A/N	Active Networks	",
    	108	:"IPComp	IP Payload Compression Protocol	RFC 3173",
    	109	:"SNP	Sitara Networks Protocol	",
    	110	:"Compaq-Peer	Compaq Peer Protocol	",
    	111	:"IPX-in-IP	IPX in IP	",
    	112	:"VRRP	Virtual Router Redundancy Protocol, Common Address Redundancy Protocol (not IANA assigned)	VRRP:RFC 3768",
    	113	:"PGM	PGM Reliable Transport Protocol	RFC 3208",
    	114	:"	Any 0-hop protocol	",
    	115	:"L2TP	Layer Two Tunneling Protocol Version 3	RFC 3931",
    	116	:"DDX	D-II Data Exchange (DDX)	",
    	117	:"IATP	Interactive Agent Transfer Protocol	",
    	118	:"STP	Schedule Transfer Protocol	",
    	119	:"SRP	SpectraLink Radio Protocol	",
    	120	:"UTI	Universal Transport Interface Protocol	",
    	121	:"SMP	Simple Message Protocol	",
    	122	:"SM	Simple Multicast Protocol	draft-perlman-simple-multicast-03",
    	123	:"PTP	Performance Transparency Protocol	",
    	124	:"IS-IS over IPv4	Intermediate System to Intermediate System (IS-IS) Protocol over IPv4	RFC 1142 and RFC 1195",
    	125	:"FIRE	Flexible Intra-AS Routing Environment	",
    	126	:"CRTP	Combat Radio Transport Protocol	",
    	127	:"CRUDP	Combat Radio User Datagram	",
    	128	:"SSCOPMCE	Service-Specific Connection-Oriented Protocol in a Multilink and Connectionless Environment	ITU-T Q.2111 (1999)",
    	129	:"IPLT		",
    	130	:"SPS	Secure Packet Shield	",
    	131	:"PIPE	Private IP Encapsulation within IP	Expired I-D draft-petri-mobileip-pipe-00.txt",
    	132	:"SCTP	Stream Control Transmission Protocol	RFC 4960",
    	133	:"FC	Fibre Channel	",
    	134	:"RSVP-E2E-IGNORE	Reservation Protocol (RSVP) End-to-End Ignore	RFC 3175",
    	135	:"Mobility Header	Mobility Extension Header for IPv6	RFC 6275",
    	136	:"UDPLite	Lightweight User Datagram Protocol	RFC 3828",
    	137	:"MPLS-in-IP	Multiprotocol Label Switching Encapsulated in IP	RFC 4023, RFC 5332",
    	138	:"manet	MANET Protocols	RFC 5498",
    	139	:"HIP	Host Identity Protocol	RFC 5201",
    	140	:"Shim6	Site Multihoming by IPv6 Intermediation	RFC 5533",
    	141	:"WESP	Wrapped Encapsulating Security Payload	RFC 5840",
    	142	:"ROHC	Robust Header Compression	RFC 5856",
    	143	:"Ethernet	IPv6 Segment Routing (TEMPORARY - registered 2020-01-31, expires 2021-01-31)	",
    	144	:"Unassigned	",
    	253	:"Use for experimentation and testing	RFC 3692",
    	255	:"Reserved"
    }
    code = int(s,16)
    if code not in dictionary: 
        return f"Unknown {code}"
    return code,dictionary[code]


# Returns index of x in arr if present, else -1
def binary_search(arr, low, high, x):
 
    # Check base case
    if high >= low:
 
        mid = (high + low) // 2
 
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return -1

def print_TCP(s):
    src_port = int(s[0 : 4], 16)
    dst_port = int(s[4 : 8], 16)
    seq      = int(s[8 :16], 16)
    ack      = int(s[16:24], 16)
    print("Src port:" , src_port , "Dst port:" , dst_port , "Seq number", seq , "Ack number", ack , "Rest",s[24:] ) 
 
def print_UDP(s):
    UDP = {}
    UDP["Src port"] = int(s[0:4], 16)
    UDP["Dst port"] = int(s[4:8], 16)
    UDP["Len"     ] = int(s[8 :16], 16)
    UDP["Checksum"] = int(s[16:24], 16)
    UDP["Rest"    ] = s[24:]
    for k,v in UDP.items():
        print(k,v, end=ENDLN)
    print()
    #print("Src port:" , src_port, "Dst port:" , dst_port, "Len", len , "Checksum", checksum , "Rest",s[24:] ) 

def get_ready_valid_data(data):
	valid_data_ready_boundle = zip( data['valid'][args.start_cc:args.end_cc:args.clk_period], 
								    data['ready'][args.start_cc:args.end_cc:args.clk_period], 
									data['data' ][args.start_cc:args.end_cc:args.clk_period],
									data['tlast'][args.start_cc:args.end_cc:args.clk_period],
									range(args.start_cc,len(data['valid'][args.start_cc:args.end_cc:args.clk_period]),args.clk_period) )
	valid_data_ready         = filter(lambda x: x[0]=='1' and x[1]=='1', valid_data_ready_boundle)
	return					 list(map(lambda x: {"data": x[2], "tlast":x[3], "t":x[4]}, valid_data_ready))

def chop_in_tlast(data):
	elements = {}
	first = True
	for element in data:
		if first:
			elements["t"] 	 = element["t"]
			elements["data"] = ""
			first = False

		data = "{:0128x}".format(int(element["data"], 2))
		elements["data"] += data

		if element["tlast"] == "1":
			elements["data"] = reorder(elements["data"])
			yield elements
			del elements
			elements = {}
			first = True
			
def prepare_data(data):
	ready_valid_data = get_ready_valid_data(data)
	return chop_in_tlast(ready_valid_data)



if __name__ == "__main__":
	from vcdvcd import VCDVCD
	import re
	import argparse
	parser = argparse.ArgumentParser(description='packet analyzer from vcd')
	parser.add_argument('vcd_path'		,       type=str, default="examples/network_stack_stops.vcd",                             help='path to the vcd that refers to cmac')
	parser.add_argument('--cmac_rx'		, 		type=str, default="dut.pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_0")
	parser.add_argument('--cmac_tx'		, 		type=str, default="dut.pfm_top_i/dynamic_region/system_ila_0/inst/net_slot_1")
	parser.add_argument('--start_cc'	, 		type=int, default=0)
	parser.add_argument('--end_cc'		, 		type=int, default=None)
	parser.add_argument('--not_rx'	    ,       default=True,  action="store_false")
	parser.add_argument('--not_tx'		,       default=True,  action="store_false")
	parser.add_argument('--raw'	    	,       default=False, action="store_true")
	parser.add_argument('--ext'	    	,       default=False, action="store_true")
	
	parser.add_argument('--clk_period'	, 		type=int, default=1)
	args = parser.parse_args()

	vcd = VCDVCD(
		args.vcd_path,
		store_tvs=True,
		store_scopes=True
	)    
	
	cmac_rx_signals = {}
	cmac_tx_signals = {}
	for label, sig_name in [("data","_axis_tdata"),("valid","_axis_tvalid"),("ready","_axis_tready"), ("tlast","_axis_tlast")]:
		cmac_rx_signals[label] = vcd[re.compile(args.cmac_rx+sig_name+".*")]
		cmac_tx_signals[label] = vcd[re.compile(args.cmac_tx+sig_name+".*")]


	data_rx = prepare_data(cmac_rx_signals)
	data_tx = prepare_data(cmac_tx_signals)


	data_rx_not_finished = args.not_rx
	data_tx_not_finished = args.not_tx
	data_rx_elem_valid 	 = False
	data_tx_elem_valid 	 = False
	data_rx_elem  		 = None	
	data_tx_elem  		 = None

	while data_rx_not_finished or data_tx_not_finished:
		
		if not data_rx_elem_valid and data_rx_not_finished:
			try:
				data_rx_elem 	   = next(data_rx)
				data_rx_elem_valid = True
			except StopIteration:
				print("finish rx")
				data_rx_not_finished  = False

		if not data_tx_elem_valid and data_tx_not_finished:
			try:
				data_tx_elem 	   = next(data_tx)
				data_tx_elem_valid = True
			except StopIteration:
				print("finish tx")
				data_tx_not_finished  = False

		if not data_rx_elem_valid and not data_tx_elem_valid:
			break
		elif data_rx_elem_valid and not data_tx_elem_valid:
			print("rx", data_rx_elem["t"], " ", end="")
			print_frame(data_rx_elem["data"]) 
			data_rx_elem_valid = False
		elif data_tx_elem_valid and not data_rx_elem_valid:
			print("tx", data_tx_elem["t"], " ", end="")
			print_frame(data_tx_elem["data"]) 
			data_tx_elem_valid = False
		elif data_tx_elem["t"] > data_rx_elem["t"]:
			print("rx", data_rx_elem["t"], " ", end="")
			print_frame(data_rx_elem["data"]) 
			data_rx_elem_valid = False
		else:
			print("tx", data_tx_elem["t"], " ", end="")
			print_frame(data_tx_elem["data"]) 
			data_tx_elem_valid = False

		input()
		#you can also parse data field via https://hpd.gasmi.net/
		
	
