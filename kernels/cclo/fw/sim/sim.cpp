#include "Stream.h"
#include "Simulation.h"
#include "Axi.h"
#include <pthread.h>
#include <iostream>
#include <sstream>
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <stdint.h>
#include "reduce_sum.h"
#include "vnx.h"
#include "ccl_offload_control.h"
#include <zmqpp/zmqpp.hpp>
#include <string>
#include <jsoncpp/json/json.h>

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

void udp_packetizer(Stream<axi::Stream<ap_uint<512> > > &in, Stream<axi::Stream<ap_uint<512>, 16> > &out, Stream<axi::Stream<ap_uint<32> > > &cmd, Stream<axi::Stream<ap_uint<32> > > &sts){
    hls::stream<ap_axiu<512,0,0,0> > in_int("udp_in_pkt");
    hls::stream<ap_axiu<512,0,0,16> > out_int("udp_out_pkt"); 
    hls::stream<ap_uint<32> > cmd_int("udp_in_cmd");
    hls::stream<ap_uint<32> > sts_int("udp_out_sts"); 

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
        tmp_in_elem.last = tmp_in.last;
        in_int.write(tmp_in_elem);
        transaction_count++;
    } while(tmp_in.last == 0);
    cout << "UDP TX message received" << endl;
    for(int i=0; i<5; i++) cmd_int.write(cmd.Pop().data);
    cout << "UDP TX command received" << endl;
    //call packetizer
    vnx_packetizer(in_int, out_int, cmd_int, sts_int, 1024);
    //load result stream
    do {
        tmp_out_elem = out_int.read();
        tmp_out.last = tmp_out_elem.last;
        tmp_out.data = tmp_out_elem.data;
        tmp_out.dest = tmp_out_elem.dest;
        out.Push(tmp_out);
        transaction_count--;
    } while(transaction_count > 0);
    cout << "UDP TX message sent" << endl;
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    cout << "UDP TX status sent" << endl;
}

void udp_depacketizer(Stream<axi::Stream<ap_uint<512>, 16> > &in, Stream<axi::Stream<ap_uint<512> > > &out, Stream<axi::Stream<ap_uint<32> > > &sts){
    hls::stream<ap_axiu<512,0,0,16> > in_int("udp_in_pkt");
    hls::stream<ap_axiu<512,0,0,0> > out_int("udp_out_pkt"); 
    hls::stream<ap_uint<32> > sts_int("udp_out_sts");

    axi::Stream<ap_uint<512>, 16> tmp_in;
    axi::Stream<ap_uint<512> > tmp_out;
    ap_axiu<512,0,0,16> tmp_in_elem;
    ap_axiu<512,0,0,0> tmp_out_elem;
    axi::Stream<ap_uint<32> > tmp_sts;
    ap_uint<32> tmp_sts_elem;
    int transaction_count;

    //load op stream
    transaction_count = 0;
    do {
        tmp_in = in.Pop();
        if(transaction_count == 0) transaction_count = (tmp_in.data + 63) / 64;
        tmp_in_elem.data = tmp_in.data;
        tmp_in_elem.last = tmp_in.last;
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
        tmp_out.data = tmp_out_elem.data;
        out.Push(tmp_out);
        transaction_count--;
    } while(tmp_out_elem.last == 0);
    cout << "UDP RX message sent" << endl;
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    tmp_sts.data = sts_int.read();
    sts.Push(tmp_sts);
    cout << "UDP RX status sent" << endl;
}

//emulate an AXI Stream Switch
void axis_switch(uint32_t *cfgmem,
            Stream<axi::Stream<ap_uint<512> > > s[8],
            Stream<axi::Stream<ap_uint<512> > > m[9]){

    //detect and reply to reconfig attempt
    if(cfgmem[0] == 2){
        cout << "Switch reconfigured" << endl;
        cfgmem[0] = 0;
    }
    for(int m_idx=0; m_idx<9; m_idx++){
        unsigned int m_cfg = cfgmem[16+m_idx];
        //src = cfgmem[16+dst] and if src is 0x80000000 do nothing
        if(m_cfg != 0x80000000){
            for(int s_idx=0; s_idx<8; s_idx++){
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
                            Stream<axi::Stream<ap_uint<32>> > &d0_cmd,
                            Stream<axi::Stream<ap_uint<32>> > &d2_sts,
                            Stream<axi::Stream<ap_uint<32>> > &d2_cmd){
    uint32_t* mer = cfgmem + IRQCTRL_MER_OFFSET/4;
    uint32_t* iar = cfgmem + IRQCTRL_IAR_OFFSET/4;
    uint32_t* ier = cfgmem + IRQCTRL_IER_OFFSET/4;
    uint32_t* ipr = cfgmem + IRQCTRL_IPR_OFFSET/4;
    bool timer_irq_active = ((*timermem & TIMER_CSR_INTERRUPT_MASK) != 0);
    bool dma0_cmd_irq_active = d0_cmd.IsEmpty();
    bool dma0_sts_irq_active = !d0_sts.IsEmpty();
    bool dma2_cmd_irq_active = d2_cmd.IsEmpty();
    bool dma2_sts_irq_active = !d2_sts.IsEmpty();
    uint32_t interrupt_pending = 0;
    //while master enable and hw interrupt enable
    //cycle through interrupt sources and set interrupt pending if mask enables
    if(*mer == (IRQCTRL_MER_HARDWARE_INTERRUPT_ENABLE|IRQCTRL_MER_MASTER_ENABLE)){
        interrupt_pending |= timer_irq_active ? IRQCTRL_TIMER_ENABLE : 0;
        interrupt_pending |= dma0_cmd_irq_active ? IRQCTRL_DMA0_CMD_QUEUE_EMPTY : 0;
        interrupt_pending |= dma0_sts_irq_active ? IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY : 0;
        interrupt_pending |= dma2_cmd_irq_active ? IRQCTRL_DMA2_CMD_QUEUE_EMPTY : 0;
        interrupt_pending |= dma2_sts_irq_active ? IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY : 0;
        if(interrupt_pending & *ier){
            cout << "Interrupt controller pending interrupt " << interrupt_pending << endl;
        }
        *ipr = (interrupt_pending & *ier);
    }
    //clear interrupts when acknowledged
    if(*iar != 0){
        cout << "Interrupt controller clear interrupt" << endl;
        *ipr = *ipr & ~(*iar);
        *iar = 0;
    }
}

template <unsigned int nports>
void udp_switch(Stream<axi::Stream<ap_uint<512>, 16> > s[nports],
            Stream<axi::Stream<ap_uint<512>, 16> > m[nports]){

    axi::Stream<ap_uint<512>, 16> tmp;
    for(int s_idx=0; s_idx<nports; s_idx++){
        if(!s[s_idx].IsEmpty()){
            do{
                tmp = s[s_idx].Pop();
                m[tmp.dest].Push(tmp);
            } while(tmp.last == 0);
            stringstream ss;
            ss << "Routed UDP s" << s_idx << " to m" << tmp.dest << "\n";
            cout << ss.str();
        }
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

void sim_bd(zmqpp::socket &socket, vector<char> &devicemem, uint32_t *cfgmem) {

    Stream<axi::Stream<ap_uint<32> >, 32> host_cmd("host_cmd");
    Stream<axi::Stream<ap_uint<32> >, 32> host_sts("host_sts");
    Stream<axi::Stream<ap_uint<512> > > krnl_rx_data;
    Stream<axi::Stream<ap_uint<512> > > krnl_tx_data;
    Stream<axi::Stream<ap_uint<512>, 16 > > udp_rx_data;
    Stream<axi::Stream<ap_uint<512>, 16 > > udp_tx_data;
    Stream<axi::Stream<ap_uint<512>, 16 > > tcp_rx_data;
    Stream<axi::Stream<ap_uint<512>, 16 > > tcp_tx_data;

    Stream<axi::Command<64, 23> > dma_write_cmd_int[3];
    Stream<axi::Command<64, 23> > dma_read_cmd_int[3];
    Stream<axi::Status > dma_write_sts_int[3];
    Stream<axi::Status > dma_read_sts_int[3];
    Stream<axi::Stream<ap_uint<512> > > dma_write_data[3];

    Stream<axi::Stream<ap_uint<512> > > switch_s[8];
    Stream<axi::Stream<ap_uint<512> > > switch_m[9];

    // Dataflow functions running in parallel
    HLSLIB_DATAFLOW_INIT();
    //DMA0
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA0_RX], dma_read_cmd_int[0]);
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA0_TX], dma_write_cmd_int[0]);
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[0], dma_write_sts_int[0], dma_write_data[0]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[0], dma_read_sts_int[0], switch_s[0]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_write_sts_int[0], sts_fifos[STS_DMA0_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_read_sts_int[0], sts_fifos[STS_DMA0_RX]);
    //DMA1
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA1_RX], dma_read_cmd_int[1]);
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA1_TX], dma_write_cmd_int[1]);
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[1], dma_write_sts_int[1], switch_m[2]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[1], dma_read_sts_int[1], switch_s[1]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_write_sts_int[1], sts_fifos[STS_DMA1_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_read_sts_int[1], sts_fifos[STS_DMA1_RX]);
    //DMA2
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA2_RX], dma_read_cmd_int[2]);
    HLSLIB_FREERUNNING_FUNCTION(mb_axis_to_dma_command, cmd_fifos[CMD_DMA2_TX], dma_write_cmd_int[2]);
    HLSLIB_FREERUNNING_FUNCTION(dma_write, devicemem, dma_write_cmd_int[2], dma_write_sts_int[2], dma_write_data[2]);
    HLSLIB_FREERUNNING_FUNCTION(dma_read, devicemem, dma_read_cmd_int[2], dma_read_sts_int[2], switch_s[2]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_write_sts_int[2], sts_fifos[STS_DMA2_TX]);
    HLSLIB_FREERUNNING_FUNCTION(dma_status_to_mb_axis, dma_read_sts_int[2], sts_fifos[STS_DMA2_RX]);
    //SWITCH
    HLSLIB_FREERUNNING_FUNCTION(axis_switch, cfgmem+SWITCH_BASEADDR/4, switch_s, switch_m);
    //ARITH
    HLSLIB_FREERUNNING_FUNCTION(arithmetic, cfgmem+GPIO_TDEST_BASEADDR/4, switch_m[3], switch_m[4], switch_s[3]);
    //COMPRESS 0
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 0, switch_m[6], switch_s[5]);
    //COMPRESS 1
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 1, switch_m[7], switch_s[6]);
    //COMPRESS 2
    HLSLIB_FREERUNNING_FUNCTION(compression, cfgmem+GPIO_TDEST_BASEADDR/4, 2, switch_m[8], switch_s[7]);
    //UDP PACK/DEPACK
    HLSLIB_FREERUNNING_FUNCTION(udp_packetizer, switch_m[0], udp_tx_data, cmd_fifos[CMD_UDP_TX], sts_fifos[STS_UDP_PKT]);
    HLSLIB_FREERUNNING_FUNCTION(udp_depacketizer, udp_rx_data, dma_write_data[0], sts_fifos[STS_UDP_RX]);
    //TCP PACK/DEPACK (for now use UDP packetizer here too)
    HLSLIB_FREERUNNING_FUNCTION(udp_packetizer, switch_m[1], tcp_tx_data, cmd_fifos[CMD_TCP_TX], sts_fifos[STS_TCP_PKT]);
    HLSLIB_FREERUNNING_FUNCTION(udp_depacketizer, tcp_rx_data, dma_write_data[2], sts_fifos[STS_TCP_RX]);
    //AXI Timer
    HLSLIB_FREERUNNING_FUNCTION(timer, cfgmem+TIMER_BASEADDR/4);
    //AXI IRQ Controller
    HLSLIB_FREERUNNING_FUNCTION(interrupt_controller, cfgmem+IRQCTRL_BASEADDR/4, cfgmem+TIMER_BASEADDR/4, sts_fifos[STS_DMA0_TX], cmd_fifos[CMD_DMA0_TX], sts_fifos[STS_DMA2_TX], cmd_fifos[CMD_DMA2_TX]);
    //ZMQ to host process
    HLSLIB_FREERUNNING_FUNCTION(serve_zmq, socket, cfgmem, devicemem, sts_fifos[CMD_HOST], cmd_fifos[STS_HOST]);
    //MICROBLAZE
    HLSLIB_FREERUNNING_FUNCTION(stream_isr);
    HLSLIB_DATAFLOW_FUNCTION(run_accl);
    HLSLIB_DATAFLOW_FINALIZE();
}

int main(){
    vector<char> devicemem;
    sem_init(&mb_irq_mutex, 0, 0);

    //ZMQ for commands
    const string endpoint = "tcp://*:5555";
    // initialize the 0MQ context
    zmqpp::context context;
    // generate a pull socket
    zmqpp::socket_type type = zmqpp::socket_type::reply;
    zmqpp::socket socket(context, type);
    // bind to the socket
    socket.bind(endpoint);

    sim_bd(socket, devicemem, cfgmem);
}
