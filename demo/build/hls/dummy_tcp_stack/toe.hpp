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
#ifndef TOE_HPP_INCLUDED
#define TOE_HPP_INCLUDED

#include "./axi_utils.hpp"

const uint16_t TCP_PROTOCOL = 0x06;

// Forward declarations.
struct rtlSessionUpdateRequest;
struct rtlSessionUpdateReply;
struct rtlSessionLookupReply;
struct rtlSessionLookupRequest;

struct ipTuple
{
	ap_uint<32>	ip_address;
	ap_uint<16>	ip_port;
};

struct mmCmd
{
	ap_uint<23>	bbt;
	ap_uint<1>	type;
	ap_uint<6>	dsa;
	ap_uint<1>	eof;
	ap_uint<1>	drr;
	ap_uint<32>	saddr;
	ap_uint<4>	tag;
	ap_uint<4>	rsvd;
	mmCmd() {}
	mmCmd(ap_uint<32> addr, ap_uint<16> len)
		:bbt(len), type(1), dsa(0), eof(1), drr(1), saddr(addr), tag(0), rsvd(0) {}
	/*mm_cmd(ap_uint<32> addr, ap_uint<16> len, ap_uint<1> last)
		:bbt(len), type(1), dsa(0), eof(last), drr(1), saddr(addr), tag(0), rsvd(0) {}*/
	/*mm_cmd(ap_uint<32> addr, ap_uint<16> len, ap_uint<4> dsa)
			:bbt(len), type(1), dsa(dsa), eof(1), drr(1), saddr(addr), tag(0), rsvd(0) {}*/
};

struct mmStatus
{
	ap_uint<4>	tag;
	ap_uint<1>	interr;
	ap_uint<1>	decerr;
	ap_uint<1>	slverr;
	ap_uint<1>	okay;
};

//TODO is this required??
struct mm_ibtt_status
{
	ap_uint<4>	tag;
	ap_uint<1>	interr;
	ap_uint<1>	decerr;
	ap_uint<1>	slverr;
	ap_uint<1>	okay;
	ap_uint<22>	brc_vd;
	ap_uint<1>	eop;
};

struct openStatus
{
	ap_uint<16>	sessionID;
	bool		success;
	openStatus() {}
	openStatus(ap_uint<16> id, bool success)
		:sessionID(id), success(success) {}
};

struct appNotification
{
	ap_uint<16>			sessionID;
	ap_uint<16>			length;
	ap_uint<32>			ipAddress;
	ap_uint<16>			dstPort;
	bool				closed;
	appNotification() {}
	appNotification(ap_uint<16> id, ap_uint<16> len, ap_uint<32> addr, ap_uint<16> port)
				:sessionID(id), length(len), ipAddress(addr), dstPort(port), closed(false) {}
	appNotification(ap_uint<16> id, bool closed)
				:sessionID(id), length(0), ipAddress(0),  dstPort(0), closed(closed) {}
	appNotification(ap_uint<16> id, ap_uint<32> addr, ap_uint<16> port, bool closed)
				:sessionID(id), length(0), ipAddress(addr),  dstPort(port), closed(closed) {}
	appNotification(ap_uint<16> id, ap_uint<16> len, ap_uint<32> addr, ap_uint<16> port, bool closed)
			:sessionID(id), length(len), ipAddress(addr), dstPort(port), closed(closed) {}
};


struct appReadRequest
{
	ap_uint<16> sessionID;
	//ap_uint<16> address;
	ap_uint<16> length;
	appReadRequest() {}
	appReadRequest(ap_uint<16> id, ap_uint<16> len)
			:sessionID(id), length(len) {}
};

struct appTxMeta
{
	ap_uint<16> sessionID;
	ap_uint<16> length;
	appTxMeta() {}
	appTxMeta(ap_uint<16> id, ap_uint<16> len)
		:sessionID(id), length(len) {}
};

struct appTxRsp
{
	ap_uint<16>	sessionID;
	ap_uint<16> length;
	ap_uint<30> remaining_space;
	ap_uint<2>	error;
	appTxRsp() {}
	appTxRsp(ap_uint<16> id, ap_uint<16> len, ap_uint<30> rem_space, ap_uint<2> err)
		:sessionID(id), length(len), remaining_space(rem_space), error(err) {}
};


template <int WIDTH>
void toe(	// Data & Memory Interface
			hls::stream<net_axis<WIDTH> >&						ipRxData,
			hls::stream<mmStatus>&						rxBufferWriteStatus,
			hls::stream<mmStatus>&						txBufferWriteStatus,
			hls::stream<net_axis<WIDTH> >&						rxBufferReadData,
			hls::stream<net_axis<WIDTH> >&						txBufferReadData,
			hls::stream<net_axis<WIDTH> >&						ipTxData,
			hls::stream<mmCmd>&							rxBufferWriteCmd,
			hls::stream<mmCmd>&							rxBufferReadCmd,
			hls::stream<mmCmd>&							txBufferWriteCmd,
			hls::stream<mmCmd>&							txBufferReadCmd,
			hls::stream<net_axis<WIDTH> >&						rxBufferWriteData,
			hls::stream<net_axis<WIDTH> >&						txBufferWriteData,
			// SmartCam Interface
			hls::stream<rtlSessionLookupReply>&			sessionLookup_rsp,
			hls::stream<rtlSessionUpdateReply>&			sessionUpdate_rsp,
			//hls::stream<ap_uint<14> >&					readFinSessionId,
			hls::stream<rtlSessionLookupRequest>&		sessionLookup_req,
			hls::stream<rtlSessionUpdateRequest>&		sessionUpdate_req,
			//hls::stream<rtlSessionUpdateRequest>&		sessionInsert_req,
			//hls::stream<rtlSessionUpdateRequest>&		sessionDelete_req,
			//hls::stream<ap_uint<14> >&					writeNewSessionId,
			// Application Interface
			hls::stream<ap_uint<16> >&					listenPortReq,
			// This is disabled for the time being, due to complexity concerns
			//hls::stream<ap_uint<16> >&					appClosePortIn,
			hls::stream<appReadRequest>&					rxDataReq,
			hls::stream<ipTuple>&						openConnReq,
			hls::stream<ap_uint<16> >&					closeConnReq,
			hls::stream<appTxMeta>&						txDataReqMeta,
			hls::stream<net_axis<WIDTH> >&						txDataReq,

			hls::stream<bool>&							listenPortRsp,
			hls::stream<appNotification>&				notification,
			hls::stream<ap_uint<16> >&					rxDataRspMeta,
			hls::stream<net_axis<WIDTH> >&						rxDataRsp,
			hls::stream<openStatus>&						openConnRsp,
			hls::stream<appTxRsp>&						txDataRsp,
			//IP Address Input
			ap_uint<32>								myIpAddress,
			//statistic
			ap_uint<16>&							regSessionCount);


#endif
