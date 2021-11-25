/*******************************************************************************
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


#include "Stream.h"
#include "Simulation.h"
#include "Axi.h"
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
#include "reduce_sum.h"
#include "vnx.h"
#include "krnl_packetizer.h"
#include "ccl_offload_control.h"
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <jsoncpp/json/json.h>
#include <chrono>
#include <numeric>

// #include "fp_hp_stream_conv.h"
// #include "hp_fp_stream_conv.h"

using namespace std;
using namespace hlslib;

void mb_axis_to_dma_command(Stream<axi::Stream<ap_uint<32> > > &in, Stream<axi::Command<64, 23> > &cmd){
    axi::Command<64, 23> ret;
    axi::Stream<ap_uint<32> > tmp;

    tmp = in.Pop();
    ret.drr = tmp.data(31,31);
    ret.eof = tmp.data(30,30);
    ret.dsa = tmp.data(29,24);
    ret.type = tmp.data(23,23);
    ret.length = tmp.data(22,0);
    tmp = in.Pop();
    ret.address(31,0) = tmp.data;
    tmp = in.Pop();
    ret.address(63,32) = tmp.data;
    tmp = in.Pop();
    ret.tag = tmp.data(3,0);
    cmd.Push(ret);
    cout << "DMA command converted" << endl;

}

void dma_status_to_mb_axis(Stream<axi::Status > &sts, Stream<axi::Stream<ap_uint<32> > > &out){
    axi::Status tmp;
    axi::Stream<ap_uint<32> > ret;

    tmp = sts.Pop();
    ret.data(3,0) = tmp.tag;
    ret.data(4,4) = tmp.internalError;
    ret.data(5,5) = tmp.decodeError;
    ret.data(6,6) = tmp.slaveError;
    ret.data(7,7) = tmp.okay;
    ret.data(30,8) = tmp.bytesReceived;
    ret.data(31,31) = tmp.endOfPacket;
    out.Push(ret);
    cout << "DMA status converted" << endl;

}

void dma_read(vector<char> &mem, Stream<axi::Command<64, 23> > &cmd, Stream<axi::Status > &sts, Stream<axi::Stream<ap_uint<512> > > &rdata){
    axi::Command<64, 23> command;
    axi::Status status;
    axi::Stream<ap_uint<512> > tmp;

    command = cmd.Pop();
    stringstream ss;
    ss << "DMA Read: Command popped. length: " << command.length << " offset: " << command.address << "\n";
    cout << ss.str();
    int byte_count = 0;
    while(byte_count < command.length){
        tmp.keep = 0;
        for(int i=0; i<64 && byte_count < command.length; i++){
            tmp.data(8*(i+1)-1, 8*i) = mem.at(command.address+byte_count);
            tmp.keep(i,i) = 1;
            byte_count++;
        }
        tmp.last = (byte_count >= command.length);
        rdata.Push(tmp);
        cout << "DMA Read: Data pushed last=" << tmp.last << endl;
    }
    status.okay = 1;
    status.tag = command.tag;
    sts.Push(status);
    cout << "DMA Read: Status pushed" << endl;

}

void dma_write(vector<char> &mem, Stream<axi::Command<64, 23> > &cmd, Stream<axi::Status > &sts, Stream<axi::Stream<ap_uint<512> > > &wdata){
    axi::Command<64, 23> command;
    axi::Status status;
    axi::Stream<ap_uint<512> > tmp;

    command = cmd.Pop();
    stringstream ss;
    ss << "DMA Write: Command popped. length: " << command.length << " offset: " << command.address << "\n";
    cout << ss.str();
    int byte_count = 0;
    while(byte_count<command.length){
        tmp = wdata.Pop();
        for(int i=0; i<64; i++){
            if(tmp.keep(i,i) == 1){
                mem.at(command.address+byte_count) = tmp.data(8*(i+1)-1, 8*i);
                byte_count++;
            }
        }
        //end of packet
        if(tmp.last){
            status.endOfPacket = 1;
            break;
        }
    }
    status.okay = 1;
    status.tag = command.tag;
    status.bytesReceived = byte_count;
    sts.Push(status);
    cout << "DMA Write: Status pushed" << endl;

}

template <unsigned int INW, unsigned int OUTW>
void dwc(Stream<axi::Stream<ap_uint<INW> > > &in, Stream<axi::Stream<ap_uint<OUTW> > > &out){
    axi::Stream<ap_uint<INW> > inword;
    axi::Stream<ap_uint<OUTW> > outword;

    //3 scenarios:
    //1:N (up) conversion - read N times from input, write 1 times to output
    //N:1 (down) conversion - read 1 times from input, write N times to output
    //N:M conversion - up-conversion to least common multiple, then down-conversion
    if(INW < OUTW && OUTW%INW == 0){
        //1:N case
        outword.keep = 0;
        outword.last = 0;
        outword.data = 0;
        for(int i=0; i<OUTW/INW; i++){
            inword = in.Pop();
            outword.data((i+1)*INW-1,i*INW) = inword.data;
            outword.keep((i+1)*INW/8-1,i*INW/8) = inword.keep;
            outword.last = 1;
            if((inword.last == 1) || (inword.keep(INW/8-1,INW/8-1) != 1)) break;
        }
        out.Push(outword);
    } else if(INW > OUTW && INW%OUTW == 0){
        //N:1 case
        inword = in.Pop();
        for(int i=0; i<INW/OUTW; i++){
            outword.data = inword.data((i+1)*OUTW-1,i*OUTW);
            outword.keep = inword.keep((i+1)*OUTW/8-1,i*OUTW/8);
            //last if actually at last input read or if any previous input read is incomplete
            outword.last = (i==(INW/OUTW-1)) || (outword.keep(OUTW/8-1,OUTW/8-1) != 1);
            out.Push(outword);
            if(outword.last == 1) break;
        }
    } else{
        unsigned const int inter_width = lcm(INW, OUTW);
        Stream<axi::Stream<ap_uint<inter_width> > > inter;
        dwc<INW, inter_width>(in, inter);
        dwc<inter_width, OUTW>(inter, out);
    }
}

void arithmetic(uint32_t *cfgmem, Stream<axi::Stream<ap_uint<512> > > &op0, Stream<axi::Stream<ap_uint<512> > > &op1, Stream<axi::Stream<ap_uint<512> > > &res){
    hls::stream<ap_axiu<1024,0,0,0> > op_int("arith_op");
    hls::stream<ap_axiu<512,0,0,0> > res_int("arith_res"); 
    axi::Stream<ap_uint<512> > tmp_op0;
    axi::Stream<ap_uint<512> > tmp_op1;
    axi::Stream<ap_uint<512> > tmp_res_strm;
    ap_axiu<1024,0,0,0> tmp_op;
    ap_axiu<512,0,0,0> tmp_res;

    //load op stream
    do {
        tmp_op0 = op0.Pop();
        tmp_op1 = op1.Pop();
        tmp_op.data(511,0) = tmp_op0.data;
        tmp_op.keep(63,0) = tmp_op0.keep;
        tmp_op.data(1023,512) = tmp_op1.data;
        tmp_op.keep(127,64) = tmp_op1.keep;
        tmp_op.last = tmp_op0.last;
        op_int.write(tmp_op);
    } while(tmp_op0.last == 0);
    cout << "Arith packet received" << endl;
    //call arith
    switch(cfgmem[0]){
        case 0:
            reduce_sum_float(op_int, res_int);
            break;
        case 1:
            reduce_sum_double(op_int, res_int);
            break;
        case 2:
            reduce_sum_int32_t(op_int, res_int);
            break;
        case 3:
            reduce_sum_int64_t(op_int, res_int);
            break;
        //half precision is problematic, no default support in C++
        // case 4:
        //     stream_add<512, half>(op_int, res_int);
        //     break;
    }
    //load result stream
    cout << "Arith packet processed" << endl;
    do {
        tmp_res = res_int.read();
        tmp_res_strm.last = tmp_res.last;
        tmp_res_strm.data = tmp_res.data;
        tmp_res_strm.keep = tmp_res.keep;
        res.Push(tmp_res_strm);
    } while(tmp_res.last == 0);
    cout << "Arith packet sent" << endl;
}

void compression(uint32_t *cfgmem, int idx, Stream<axi::Stream<ap_uint<512> > > &op0, Stream<axi::Stream<ap_uint<512> > > &res){ 
    axi::Stream<ap_uint<512> > tmp_op0;
    axi::Stream<ap_uint<512> > tmp_res;

    int tdest = (cfgmem[0] >> 8*(1+idx)) & 0xff;
    if(!op0.IsEmpty()){
        cout << "Running compression on lane " << idx << " with TDEST=" << tdest << endl;
        switch(tdest){
            case 0:
                res.Push(op0.Pop());
                break;
            case 1://downcast
                tmp_op0 = op0.Pop();
                for(int i=0; i<16; i++){
                    tmp_res.data(16*(i+1)-1,16*i) = tmp_op0.data(32*(i+1)-1,32*i+16);
                }
                tmp_op0 = op0.Pop();
                for(int i=0; i<16; i++){
                    tmp_res.data(16*(i+16+1)-1,16*(i+16)) = tmp_op0.data(32*(i+1)-1,32*i+16);
                }
                res.Push(tmp_res);
                break;
            case 2://upcast
                tmp_op0 = op0.Pop();
                tmp_res.data = 0;
                for(int i=0; i<16; i++){
                    tmp_res.data(32*(i+1)-1,32*i+16) = tmp_op0.data(16*(i+1)-1,16*i);
                }
                res.Push(tmp_res);
                tmp_res.data = 0;
                for(int i=0; i<16; i++){
                    tmp_res.data(32*(i+1)-1,32*i) = tmp_op0.data(16*(i+16+1)-1,16*(i+16));
                }
                res.Push(tmp_res);
                break;
        }
    }

}

void ext_kernel_packetizer(Stream<axi::Stream<ap_uint<512> > > &in, Stream<axi::Stream<ap_uint<512> > > &out, Stream<axi::Stream<ap_uint<32> > > &cmd, Stream<axi::Stream<ap_uint<32> > > &sts){
    hls::stream<ap_axiu<512,0,0,0> > in_int("krnl_in_pkt");
    hls::stream<ap_axiu<512,0,0,0> > out_int("krnl_out_pkt"); 
    hls::stream<ap_uint<32> > cmd_int("krnl_in_cmd");
    hls::stream<ap_uint<32> > sts_int("krnl_out_sts"); 

    axi::Stream<ap_uint<512> > tmp_in;
    axi::Stream<ap_uint<512> > tmp_out;
    ap_axiu<512,0,0,0> tmp_out_elem;
    ap_axiu<512,0,0,0> tmp_in_elem;
    axi::Stream<ap_uint<32> > tmp_sts;
    ap_uint<32> tmp_sts_elem;
    int transaction_count;

    //load op stream
    transaction_count = 0;
    do {
        tmp_in = in.Pop();
        tmp_in_elem.data = tmp_in.data;
        tmp_in_elem.keep = tmp_in.keep;
        tmp_in_elem.last = tmp_in.last;
        in_int.write(tmp_in_elem);
        transaction_count++;
    } while(tmp_in.last == 0);
    cmd_int.write(cmd.Pop().data);
    //call packetizer
    krnl_packetizer(in_int, out_int, cmd_int, sts_int);
    //load result stream
    do {
        tmp_out_elem = out_int.read();
        tmp_out.last = tmp_out_elem.last;
        tmp_out.data = tmp_out_elem.data;
        tmp_out.keep = tmp_out_elem.keep;
        out.Push(tmp_out);
        transaction_count--;
    } while(transaction_count > 0);
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
}

void udp_packetizer(Stream<axi::Stream<ap_uint<512> > > &in, Stream<axi::Stream<ap_uint<512>, 16> > &out, Stream<axi::Stream<ap_uint<32> > > &cmd, Stream<axi::Stream<ap_uint<32> > > &sts){
    hls::stream<ap_axiu<512,0,0,0> > in_int("udp_in_pkt");
    hls::stream<ap_axiu<512,0,0,16> > out_int("udp_out_pkt"); 
    hls::stream<ap_uint<32> > cmd_int("udp_cmd_pkt");
    hls::stream<ap_uint<32> > sts_int("udp_sts_pkt"); 

    axi::Stream<ap_uint<512> > tmp_in;
    axi::Stream<ap_uint<512>, 16> tmp_out;
    ap_axiu<512,0,0,16> tmp_out_elem;
    ap_axiu<512,0,0,0> tmp_in_elem;
    axi::Stream<ap_uint<32> > tmp_sts;
    ap_uint<32> tmp_sts_elem;
    int transaction_count;

    //load op stream
    transaction_count = 0;
    do {
        tmp_in = in.Pop();
        tmp_in_elem.data = tmp_in.data;
        tmp_in_elem.keep = tmp_in.keep;
        tmp_in_elem.last = tmp_in.last;
        in_int.write(tmp_in_elem);
        transaction_count++;
    } while(tmp_in.last == 0);
    cout << "UDP TX message received" << endl;
    for(int i=0; i<6; i++) cmd_int.write(cmd.Pop().data);
    cout << "UDP TX command received" << endl;
    //call packetizer
    vnx_packetizer(in_int, out_int, cmd_int, sts_int, 1024);
    //load result stream
    transaction_count++;//inc transaction count because of the 64B header
    do {
        tmp_out_elem = out_int.read();
        tmp_out.last = tmp_out_elem.last;
        tmp_out.data = tmp_out_elem.data;
        tmp_out.keep = tmp_out_elem.keep;
        tmp_out.dest = tmp_out_elem.dest;
        out.Push(tmp_out);
        transaction_count--;
    } while(transaction_count > 0);
    cout << "UDP TX message sent" << endl;
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    cout << "UDP TX status sent" << endl;
}

void udp_depacketizer(Stream<axi::Stream<ap_uint<512>, 16> > &in, Stream<axi::Stream<ap_uint<512>, 16> > &out, Stream<axi::Stream<ap_uint<4*32> > > &sts){
    hls::stream<ap_axiu<512,0,0,16> > in_int("udp_in_dpkt");
    hls::stream<ap_axiu<512,0,0,16> > out_int("udp_out_dpkt"); 
    hls::stream<ap_axiu<4*32,0,0,0> > sts_int("udp_sts_dpkt");

    axi::Stream<ap_uint<512>, 16> tmp_in;
    axi::Stream<ap_uint<512>, 16> tmp_out;
    ap_axiu<512,0,0,16> tmp_in_elem;
    ap_axiu<512,0,0,16> tmp_out_elem;
    axi::Stream<ap_uint<4*32> > tmp_sts;
    ap_axiu<4*32,0,0,0> tmp_sts_elem;
    unsigned int transaction_count;

    //load op stream
    transaction_count = 0;
    do {
        tmp_in = in.Pop();
        if(transaction_count == 0) {
            transaction_count = (tmp_in.data(31,0) + 63) / 64;
            transaction_count++; //increment to account for the header
        }
        stringstream ss;
        ss << "UDP RX transaction count " << transaction_count << "\n";
        cout << ss.str();
        tmp_in_elem.data = tmp_in.data;
        tmp_in_elem.last = tmp_in.last;
        tmp_in_elem.keep = tmp_in.keep;
        tmp_in_elem.dest = tmp_in.dest;
        in_int.write(tmp_in_elem);
        transaction_count--;
    } while(transaction_count > 0);
    cout << "UDP RX message received" << endl;
    //call packetizer
    vnx_depacketizer(in_int, out_int, sts_int);
    //load result stream
    do {
        tmp_out_elem = out_int.read();
        tmp_out.last = tmp_out_elem.last;
        tmp_out.keep = tmp_out_elem.keep;
        tmp_out.data = tmp_out_elem.data;
        tmp_out.dest = tmp_out_elem.dest;
        out.Push(tmp_out);
    } while(tmp_out_elem.last == 0);
    cout << "UDP RX message sent" << endl;
    if(tmp_out.dest == 0){
        tmp_sts_elem = sts_int.read();
        tmp_sts.data = tmp_sts_elem.data;
        tmp_sts.keep = tmp_sts_elem.keep;
        tmp_sts.last = tmp_sts_elem.last;
        sts.Push(tmp_sts);
        cout << "UDP RX status sent" << endl;
    }
}

//strip TDEST from a stream
template <unsigned int DWIDTH, unsigned int DESTWIDTH>
void strip_tdest( Stream<axi::Stream<ap_uint<DWIDTH>, DESTWIDTH> > &in,
                        Stream<axi::Stream<ap_uint<DWIDTH> > > &out){
    axi::Stream<ap_uint<DWIDTH>, DESTWIDTH> tmpin;
    axi::Stream<ap_uint<DWIDTH> > tmpout;
    tmpin = in.Pop();
    tmpout.data = tmpin.data;
    tmpout.last = tmpin.last;
    tmpout.keep = tmpin.keep;
    out.Push(tmpout);
}

//emulate an AXI Stream Switch with TDEST routing
template <unsigned int NSLAVES, unsigned int NMASTERS, unsigned int DWIDTH, unsigned int DESTWIDTH>
void axis_switch_tdest( Stream<axi::Stream<ap_uint<DWIDTH>, DESTWIDTH> > s[NSLAVES],
                        Stream<axi::Stream<ap_uint<DWIDTH>, DESTWIDTH> > m[NMASTERS]){

    axi::Stream<ap_uint<DWIDTH>, DESTWIDTH> word;
    for(int i=0; i<NSLAVES; i++){
        if(!s[i].IsEmpty()){
            do{
                word = s[i].Pop();
                m[min(NMASTERS-1, (unsigned int)word.dest)].Push(word);
            } while(word.last == 0);
        }
    }
}

//emulate an AXI Stream Switch with register routing
void axis_switch(uint32_t *cfgmem,
            Stream<axi::Stream<ap_uint<512> > > s[7],
            Stream<axi::Stream<ap_uint<512> > > m[8]){

    //detect and reply to reconfig attempt
    if(cfgmem[0] == 2){
        cout << "Switch reconfigured" << endl;
        cfgmem[0] = 0;
    }
    for(int m_idx=0; m_idx<8; m_idx++){
        unsigned int m_cfg = cfgmem[16+m_idx];
        //src = cfgmem[16+dst] and if src is 0x80000000 do nothing
        if(m_cfg != 0x80000000){
            for(int s_idx=0; s_idx<7; s_idx++){
                if(s_idx == m_cfg){
                    if(!s[s_idx].IsEmpty()){
                        m[m_idx].Push(s[s_idx].Pop());
                        stringstream ss;
                        ss << "Routing s" << s_idx << " to m" << m_idx << "\n";
                        cout << ss.str();
                    }
                }
            }
        }
    }
}

//emulate the required features of an AXI Timer (PG079)
void timer(uint32_t *cfgmem){
    uint32_t* csr = cfgmem + TIMER_CSR0_OFFSET/4;
    uint32_t* lr = cfgmem + TIMER_LR0_OFFSET/4;
    uint32_t* cr = cfgmem + TIMER_CR0_OFFSET/4;
    if(*csr & TIMER_CSR_ENABLE_ALL_MASK){
        cout << "Timer enable all" << endl;
        *csr |= TIMER_CSR_ENABLE_MASK;
    }
    if(*csr & TIMER_CSR_ENABLE_MASK){
        if(*csr & TIMER_CSR_LOAD_TIMER_MASK){
            *cr = *lr;
        } else if(*csr & TIMER_CSR_UP_DOWN_MASK){
            *cr--;
        } else {
            *cr++;
        }
        if(*cr == 0){
            cout << "Timer set interrupt" << endl;
            *csr |= TIMER_CSR_INTERRUPT_MASK;
        }
    } else if(*csr & TIMER_CSR_INTERRUPT_MASK){
        cout << "Timer clear interrupt" << endl;
        *csr &= ~TIMER_CSR_INTERRUPT_MASK;
    }
}

//emulate interrupt controller
//we have the following sources of interrupts:
//irq0 - empty command stream to DMA0/DMA2
//irq1 - non-empty status stream from DMA0/DMA2
//irq2 - timer
void interrupt_controller(uint32_t *cfgmem, uint32_t *timermem, 
                            Stream<axi::Stream<ap_uint<32>> > &d0_sts,
                            Stream<axi::Stream<ap_uint<32>> > &d0_cmd){
    uint32_t* mer = cfgmem + IRQCTRL_MER_OFFSET/4;
    uint32_t* iar = cfgmem + IRQCTRL_IAR_OFFSET/4;
    uint32_t* ier = cfgmem + IRQCTRL_IER_OFFSET/4;
    uint32_t* ipr = cfgmem + IRQCTRL_IPR_OFFSET/4;
    bool timer_irq_active = ((*timermem & TIMER_CSR_INTERRUPT_MASK) != 0);
    bool dma0_cmd_irq_active = d0_cmd.IsEmpty();
    bool dma0_sts_irq_active = !d0_sts.IsEmpty();
    uint32_t interrupt_pending = 0;
    uint32_t prev_interrupt_pending = 0;
    //while master enable and hw interrupt enable
    //cycle through interrupt sources and set interrupt pending if mask enables
    if(*mer == (IRQCTRL_MER_HARDWARE_INTERRUPT_ENABLE|IRQCTRL_MER_MASTER_ENABLE)){
        interrupt_pending |= timer_irq_active ? IRQCTRL_TIMER_ENABLE : 0;
        interrupt_pending |= dma0_cmd_irq_active ? IRQCTRL_DMA0_CMD_QUEUE_EMPTY : 0;
        interrupt_pending |= dma0_sts_irq_active ? IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY : 0;

        if(interrupt_pending & *ier){
            if(prev_interrupt_pending != interrupt_pending){
                cout << "Interrupt controller pending interrupt " << interrupt_pending << endl;
            }
        }
        *ipr = (interrupt_pending & *ier);
        prev_interrupt_pending = interrupt_pending;
    }
    //clear interrupts when acknowledged
    if(*iar != 0){
        cout << "Interrupt controller clear interrupt " << *iar << endl;
        *ipr = *ipr & ~(*iar);
        *iar = 0;
    }
}

void serve_zmq(zmqpp::socket &socket, uint32_t *cfgmem, vector<char> &devicemem, Stream<axi::Stream<ap_uint<32>> > &cmd, Stream<axi::Stream<ap_uint<32>> > &sts){

    Json::Reader reader;
    Json::StreamWriterBuilder builder;

    // receive the message
    zmqpp::message message;
    // decompose the message 
    socket.receive(message);
    string msg_text;
    message >> msg_text;//message now is in a string

#ifdef ZMQ_CALL_VERBOSE
    cout << "Received: " << msg_text << endl;
#endif

    //parse msg_text as json
    Json::Value request;
    reader.parse(msg_text, request); // reader can also read strings

    //parse message and reply
    Json::Value response;
    response["status"] = 0;
    int adr, val, len;
    uint64_t dma_addr;
    Json::Value dma_wdata;
    switch(request["type"].asUInt()){
        // MMIO read request  {"type": 0, "addr": <uint>}
        // MMIO read response {"status": OK|ERR, "rdata": <uint>}
        case 0:
            adr = request["addr"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "MMIO read " << adr << endl;
#endif
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
                response["rdata"] = 0;
            } else {
                response["rdata"] = cfgmem[adr/4];
            }
            break;
        // MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
        // MMIO write response {"status": OK|ERR}
        case 1:
            adr = request["addr"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "MMIO write " << adr << endl;
#endif
            if(adr >= END_OF_EXCHMEM){
                response["status"] = 1;
            } else {
                cfgmem[adr/4] = request["wdata"].asUInt();
            }
            break;
        // Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
        // Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
        case 2:
            adr = request["addr"].asUInt();
            len = request["len"].asUInt();
#ifdef ZMQ_CALL_VERBOSE
            cout << "Mem read " << adr << " len: " << len << endl;
#endif
            if((adr+len) > devicemem.size()){
                response["status"] = 1;
                response["rdata"][0] = 0;
            } else {
                for (int i=0; i<len; i++) 
                { 
                    response["rdata"][i] = devicemem.at(adr+i);
                }
            }
            break;
        // Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>}
        // Devicemem write response {"status": OK|ERR}
        case 3:
            adr = request["addr"].asUInt();
            dma_wdata = request["wdata"];
            len = dma_wdata.size();
#ifdef ZMQ_CALL_VERBOSE
            cout << "Mem write " << adr << " len: " << len << endl;
#endif
            if((adr+len) > devicemem.size()){
                devicemem.resize(adr+len);
            }
            for(int i=0; i<len; i++){
                devicemem.at(adr+i) = dma_wdata[i].asUInt();
            }
            break;
        // Call request  {"type": 4, arg names and values}
        // Call response {"status": OK|ERR}
        case 4:
#ifdef ZMQ_CALL_VERBOSE
            cout << "Call with scenario " << request["scenario"].asUInt() << endl;
#endif
            cmd.Push(axi::Stream<ap_uint<32> >(request["scenario"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["count"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["comm"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["root_src_dst"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["function"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["tag"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["arithcfg"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["compression_flags"].asUInt()));
            cmd.Push(axi::Stream<ap_uint<32> >(request["stream_flags"].asUInt()));
            dma_addr = request["addr_0"].asUInt64();
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr & 0xffffffff)));
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr >> 32)));
            dma_addr = request["addr_1"].asUInt64();
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr & 0xffffffff)));
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr >> 32)));
            dma_addr = request["addr_2"].asUInt64();
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr & 0xffffffff)));
            cmd.Push(axi::Stream<ap_uint<32> >((uint32_t)(dma_addr >> 32)));
            //pop the status queue to wait for call completion
            sts.Pop();
            break;
        default:
#ifdef ZMQ_CALL_VERBOSE
            cout << "Unrecognized message" << endl;
#endif
            response["status"] = 1;
    }
    //return message to client
    string str = Json::writeString(builder, response);
    socket.send(str);
}

void eth_endpoint_egress_port(zmqpp::socket &socket, Stream<axi::Stream<ap_uint<512>, 16> > &in){

    zmqpp::message message;
    Json::Value packet;
    Json::StreamWriterBuilder builder;

    //pop first word in packet
    stringstream ss;
    unsigned int dest;
    axi::Stream<ap_uint<512>, 16> tmp;
    //get the data (bytes valid from tkeep)
    unsigned int idx=0;
    do{
        tmp = in.Pop();
        for(int i=0; i<64; i++){ 
            if(tmp.keep(i,i) == 1){
                packet["data"][idx++] = (unsigned int)tmp.data(8*(i+1)-1,8*i);
            }
        }
    }while(tmp.last == 0);
    dest = tmp.dest;
    //first part of the message is the destination port ID
    message << to_string(dest);
    //finally package the data
    string str = Json::writeString(builder, packet);
    message << str;
    cout << "ETH Send to " << dest << " " << str;
    socket.send(message);
}

void eth_endpoint_ingress_port(zmqpp::socket &socket, Stream<axi::Stream<ap_uint<512>, 16> > &out){
    
    Json::Reader reader;

    // receive the message
    zmqpp::message message;
    // decompose the message 
    socket.receive(message);
    string msg_text, dst_text, src_text;

    //get and check destination ID
    message >> dst_text;
    message >> msg_text;
    cout << "ETH Receive " << msg_text;

    //parse msg_text as json
    Json::Value packet, data;
    reader.parse(msg_text, packet);

    data = packet["data"];
    unsigned int len = data.size();

    axi::Stream<ap_uint<512>, 16> tmp;
    int idx = 0;
    while(idx<len){
        for(int i=0; i<64; i++){
            if(idx<len){
                tmp.data(8*(i+1)-1,8*i) = data[idx++].asUInt();
                tmp.keep(i,i) = 1;
            } else{
                tmp.keep(i,i) = 0;
            }
        }
        tmp.last = (idx == len);
        out.Push(tmp);
    }
}

void dummy_external_kernel(Stream<axi::Stream<ap_uint<512>, 16> > &in, Stream<axi::Stream<ap_uint<512> > > &out){
    axi::Stream<ap_uint<512>, 16> tmp;
    axi::Stream<ap_uint<512> > tmp_no_tdest;
    tmp = in.Pop();
    stringstream ss;
    ss << "External Kernel Interface: Read TDEST=" << tmp.dest << "\n";
    cout << ss.str();
    tmp_no_tdest.data = tmp.data;
    tmp_no_tdest.last = tmp.last;
    tmp_no_tdest.keep = tmp.keep;
    out.Push(tmp_no_tdest);
}

void sim_bd(zmqpp::socket &cmd_socket, 
            zmqpp::socket &eth_tx_socket, 
            zmqpp::socket &eth_rx_socket, 
            vector<char> &devicemem, 
            uint32_t *cfgmem) {

    Stream<axi::Stream<ap_uint<32> >, 32> host_cmd("host_cmd");
    Stream<axi::Stream<ap_uint<32> >, 32> host_sts("host_sts");
    Stream<axi::Stream<ap_uint<512> >, 32> krnl_to_accl_data;
    Stream<axi::Stream<ap_uint<512>, 16> > accl_to_krnl_data[1];
    Stream<axi::Stream<ap_uint<512>, 16> > eth_rx_data;
    Stream<axi::Stream<ap_uint<512>, 16> > eth_rx_data_int[1];
    Stream<axi::Stream<ap_uint<512>, 16> > eth_rx_data_switched[2];
    Stream<axi::Stream<ap_uint<512>, 16> > eth_tx_data;

    Stream<axi::Command<64, 23> > dma_write_cmd_int[2];
    Stream<axi::Command<64, 23> > dma_read_cmd_int[2];
    Stream<axi::Status > dma_write_sts_int[2];
    Stream<axi::Status > dma_read_sts_int[2];
    Stream<axi::Stream<ap_uint<512> > > dma_write_data[2];

    Stream<axi::Stream<ap_uint<512> > > switch_s[7];
    Stream<axi::Stream<ap_uint<512> > > switch_m[8];

    Stream<axi::Stream<ap_uint<4*32> > > eth_rx_status;
    Stream<axi::Stream<ap_uint<4*32> > > tcp_rx_status;

    // Dataflow functions running in parallel
    HLSLIB_DATAFLOW_INIT();
    //DMA0
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA0_RX], dma_read_cmd_int[0]);
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA0_TX], dma_write_cmd_int[0]);
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[0], dma_write_sts_int[0], dma_write_data[0]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[0], dma_read_sts_int[0], switch_s[SWITCH_S_DMA0_RX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_write_sts_int[0], sts_fifos[STS_DMA0_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_read_sts_int[0], sts_fifos[STS_DMA0_RX]);
    //DMA1
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA1_RX], dma_read_cmd_int[1]);
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA1_TX], dma_write_cmd_int[1]);
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[1], dma_write_sts_int[1], switch_m[SWITCH_M_DMA1_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[1], dma_read_sts_int[1], switch_s[SWITCH_S_DMA1_RX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_write_sts_int[1], sts_fifos[STS_DMA1_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_read_sts_int[1], sts_fifos[STS_DMA1_RX]);
    //SWITCH
    HLSLIB_FREERUNNING_FUNCTION(axis_switch, cfgmem+SWITCH_BASEADDR/4, switch_s, switch_m);
    //ARITH
    HLSLIB_FREERUNNING_FUNCTION(arithmetic, cfgmem+GPIO_TDEST_BASEADDR/4, switch_m[SWITCH_M_ARITH_OP0], switch_m[SWITCH_M_ARITH_OP1], switch_s[SWITCH_S_ARITH_RES]);
    //COMPRESS 0
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 0, switch_m[SWITCH_M_COMPRESS0], switch_s[SWITCH_S_COMPRESS0]);
    //COMPRESS 1
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 1, switch_m[SWITCH_M_COMPRESS1], switch_s[SWITCH_S_COMPRESS1]);
    //COMPRESS 2
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 2, switch_m[SWITCH_M_COMPRESS2], switch_s[SWITCH_S_COMPRESS2]);
    //network PACK/DEPACK
    //TODO: implement conditional instantiation of UDP or TCP (de)packetizer here
    HLSLIB_FREERUNNING_FUNCTION(udp_packetizer, switch_m[SWITCH_M_NET_TX], eth_tx_data, cmd_fifos[CMD_NET_TX], sts_fifos[STS_NET_PKT]);
    HLSLIB_FREERUNNING_FUNCTION(udp_depacketizer, eth_rx_data, eth_rx_data_int[0], eth_rx_status);
    HLSLIB_FREERUNNING_FUNCTION(dwc<128,32>, eth_rx_status, sts_fifos[STS_NET_RX]);
    //TDEST switch to streaming kernel output and the DMA
    HLSLIB_FREERUNNING_FUNCTION(axis_switch_tdest<1,2,512,16>, eth_rx_data_int, eth_rx_data_switched);
    HLSLIB_FREERUNNING_FUNCTION(strip_tdest<512,16>, eth_rx_data_switched[0], dma_write_data[0]);
    HLSLIB_FREERUNNING_FUNCTION(ext_kernel_packetizer, krnl_to_accl_data, switch_s[SWITCH_S_EXT_KRNL], cmd_fifos[CMD_KRNL_PKT], sts_fifos[STS_KRNL_PKT]);
    HLSLIB_FREERUNNING_FUNCTION(dummy_external_kernel, eth_rx_data_switched[1], krnl_to_accl_data);
    //AXI Timer
    HLSLIB_FREERUNNING_FUNCTION(timer, cfgmem+TIMER_BASEADDR/4);
    //AXI IRQ Controller
    HLSLIB_FREERUNNING_FUNCTION(interrupt_controller, cfgmem+IRQCTRL_BASEADDR/4, cfgmem+TIMER_BASEADDR/4, sts_fifos[STS_DMA0_TX], cmd_fifos[CMD_DMA0_TX]);
    //ZMQ to host process
    HLSLIB_FREERUNNING_FUNCTION(serve_zmq, cmd_socket, cfgmem, devicemem, sts_fifos[CMD_HOST], cmd_fifos[STS_HOST]);
    //ZMQ to other nodes process(es)
    HLSLIB_FREERUNNING_FUNCTION(eth_endpoint_egress_port, eth_tx_socket, eth_tx_data);
    HLSLIB_FREERUNNING_FUNCTION(eth_endpoint_ingress_port, eth_rx_socket, eth_rx_data);
    //MICROBLAZE
    HLSLIB_FREERUNNING_FUNCTION(stream_isr);
    HLSLIB_DATAFLOW_FUNCTION(run_accl);
    HLSLIB_DATAFLOW_FINALIZE();
}

int main(int argc, char** argv){
    vector<char> devicemem;
    sem_init(&mb_irq_mutex, 0, 0);

    unsigned int world_size = atoi(argv[1]);
    unsigned int local_rank = atoi(argv[2]);
    const string endpoint_base = "tcp://127.0.0.1:";
    unsigned int starting_port = atoi(argv[3]);
    string cmd_endpoint = endpoint_base + to_string(starting_port + local_rank);
    cout << cmd_endpoint << endl;
    vector<string> eth_endpoints;

    for(int i=0; i<world_size; i++){
        eth_endpoints.emplace_back(endpoint_base + to_string(starting_port+world_size+i));
        cout << eth_endpoints.at(i) << endl;
    }
    
    //ZMQ for commands
    // initialize the 0MQ context
    zmqpp::context context;
    zmqpp::socket cmd_socket(context, zmqpp::socket_type::reply);
    zmqpp::socket eth_tx_socket(context, zmqpp::socket_type::pub);
    zmqpp::socket eth_rx_socket(context, zmqpp::socket_type::sub);
    // bind to the socket(s)
    cout << "Rank " << local_rank << " binding to " << cmd_endpoint << " and " << eth_endpoints.at(local_rank) << endl;
    cmd_socket.bind(cmd_endpoint);
    eth_tx_socket.bind(eth_endpoints.at(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    // connect to the sockets
    for(int i=0; i<world_size; i++){
        cout << "Rank " << local_rank << " connecting to " << eth_endpoints.at(i) << endl;
        eth_rx_socket.connect(eth_endpoints.at(i));
    }

    this_thread::sleep_for(chrono::milliseconds(1000));

    cout << "Rank " << local_rank << " subscribing to " << local_rank << endl;
    eth_rx_socket.subscribe(to_string(local_rank));

    this_thread::sleep_for(chrono::milliseconds(1000));

    sim_bd(cmd_socket, eth_tx_socket, eth_rx_socket, devicemem, cfgmem);
}
