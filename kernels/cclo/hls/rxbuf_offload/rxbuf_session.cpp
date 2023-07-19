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

#include "rxbuf_offload.h"
#include "Axi.h"

using namespace std;

void rxbuf_session_status(
    STREAM<rxbuf_status_control> &cmd_in,
    STREAM<ap_uint<32> > &sts_in,
    STREAM<ap_uint<32> > &sts_out
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off
    static ap_uint<32> mem[512];//TODO make depth compile-time configurable
    hlslib::axi::Status tmp_sts, stored_sts;
    rxbuf_status_control cmd;
    if(!STREAM_IS_EMPTY(cmd_in)){
        cmd = STREAM_READ(cmd_in);
        tmp_sts = hlslib::axi::Status(STREAM_READ(sts_in));
        if(cmd.first && cmd.last){
            STREAM_WRITE(sts_out, tmp_sts);
        } else if(cmd.first){
            mem[cmd.index] = tmp_sts;
        } else {
            stored_sts = hlslib::axi::Status(mem[cmd.index]);
            stored_sts.bytesReceived += tmp_sts.bytesReceived;
            stored_sts.internalError |= tmp_sts.internalError;
            stored_sts.decodeError |= tmp_sts.decodeError;
            stored_sts.slaveError |= tmp_sts.slaveError;
            stored_sts.okay &= tmp_sts.okay;
            stored_sts.endOfPacket |= tmp_sts.endOfPacket;
            mem[cmd.index] = stored_sts;
            if(cmd.last){
                STREAM_WRITE(sts_out, stored_sts);
            }                
        }
    }
}

void rxbuf_session_command(
	STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &rxbuf_dma_cmd, 
	STREAM<ap_uint<32> > &rxbuf_idx_in,
    STREAM<ap_uint<32> > &rxbuf_idx_out,
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &fragment_dma_cmd,
    STREAM<eth_notification> &session_notification,
    STREAM<eth_header> &eth_hdr_in,
    STREAM<eth_header> &eth_hdr_out,
    STREAM<rxbuf_status_control> &status_instruction
){
#pragma HLS PIPELINE II=1 style=flp
#pragma HLS INLINE off
    static rxbuf_session_descriptor mem[512];//TODO make depth compile-time configurable
    eth_notification notif;
    rxbuf_session_descriptor desc;
    hlslib::axi::Command<64, 23> cmd;
    ap_axiu<104,0,0,DEST_WIDTH> cmd_word;
    stream_word inword;
    eth_header hdr;
    rxbuf_status_control sts_command;
    if(!STREAM_IS_EMPTY(session_notification)){
        notif = STREAM_READ(session_notification);
        desc = mem[notif.session_id];
        if(desc.active){
            //descriptor exists, so this is not a SOM but rather a SOF or EOF
            //if SOF issue command to datamover
            if(notif.type == 1){
                cmd.address = desc.address;
                cmd.length = notif.length;
                cmd_word.data = cmd;
                cmd_word.last = 1;//always last, each command is a single word
                cmd_word.dest = 0;//always write RX data to device (not host)
                STREAM_WRITE(fragment_dma_cmd, cmd_word);
            } else {
                //if EOF update address in descriptor
                desc.address += notif.length;
                desc.remaining -= notif.length;
                //command the status parser
                sts_command.index = notif.session_id;
                sts_command.last = (desc.remaining == 0);
                STREAM_WRITE(status_instruction, sts_command);
                //if remaining is zero, flush
                if(desc.remaining == 0){
                    STREAM_WRITE(rxbuf_idx_out, desc.buf_index);
                    STREAM_WRITE(eth_hdr_out, desc.header);
                    desc.active = false;
                }
            }
            //prime the command to status parser
            sts_command.first = false;
        } else {
            //this is a SOM notification so descriptor does not exist, initialize
            auto rxbuf_dma_cmd_word = STREAM_READ(rxbuf_dma_cmd);
            cmd = hlslib::axi::Command<64, 23>(rxbuf_dma_cmd_word.data);//{undex, current write address, bytes remaining, header}
            desc.active = true;
            desc.buf_index = STREAM_READ(rxbuf_idx_in);
            desc.mem_index = rxbuf_dma_cmd_word.dest;
            desc.address = cmd.address;
            //store header
	        desc.header = STREAM_READ(eth_hdr_in);
            desc.remaining = desc.header.count;
            //prime the command to status parser
            sts_command.first = true;
        }
        //store descriptor
        mem[notif.session_id] = desc;
    }
}

void rxbuf_session(
    //interface to enqueue/dequeue engines; we intercept the DMA and inflight streams
	STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &rxbuf_dma_cmd, //incoming command for a full DMA transfer targeting a RX buffer
    STREAM<ap_uint<32> > &rxbuf_dma_sts, //status of a rxbuf write (assembled from statuses of fragments)
	STREAM<ap_uint<32> > &rxbuf_idx_in, //indicates RX buffer index corresponding to DMA command on rxbuf_dma_cmd
    STREAM<ap_uint<32> > &rxbuf_idx_out, //forward index of completed RX buffer
    //interface to Datamover
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &fragment_dma_cmd, //outgoing DMA command for a partial message write to a RX buffer
	STREAM<ap_uint<32> > &fragment_dma_sts, //status of a fragment write
    //interface to depacketizer
    STREAM<eth_notification> &session_notification, //get notified when there is data for a session
    STREAM<eth_header> &eth_hdr_in,
    //status to dequeuer
    STREAM<eth_header> &eth_hdr_out //forward header of message in completed RX buffer
){
#pragma HLS INTERFACE axis register both port=rxbuf_dma_cmd
#pragma HLS INTERFACE axis register both port=rxbuf_dma_sts
#pragma HLS INTERFACE axis register both port=rxbuf_idx_in
#pragma HLS INTERFACE axis register both port=rxbuf_idx_out
#pragma HLS INTERFACE axis register both port=fragment_dma_cmd
#pragma HLS INTERFACE axis register both port=fragment_dma_sts
#pragma HLS INTERFACE axis register both port=session_notification
#pragma HLS INTERFACE axis register both port=eth_hdr_in
#pragma HLS INTERFACE axis register both port=eth_hdr_out
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW disable_start_propagation
    //how this works:
    //
    //each session carries messages consecutively, so we have at any moment one active spare buffer per session
    //this means we can only have as many active sessions as there are spare buffers
    //internal storage size is ~(header length + address length + 2*count length)*min(max_num_spare_buffers, max active sessions)
    //address could be procedurally generated from a base offset, max len, and a buffer index, but still quite a lot of data to store
    //so <= 64 sessions/buffers for cheap storage in LUTRAM, <= 1024/4096 sessions/buffers for more expensive storage in BRAM/URAM
    //
    //{session_id, fragment length} <= notification
    //we query internal data store with the session id to find the active buffer descriptor
    //if there's a hit in the data store, 
    //  we get the current active spare buffer descriptor from storage {index, current write address, buf length, bytes remaining, header}
    //  we emit a DMA command for the fragment on fragment_dma_cmd
    //  we update the current spare buffer address by adding fragment length to it
    //  we copy length bytes from data_in to data_out
    //  we update the current bytes remaining by subtracting fragment length
    //  if bytes remaining is zero
    //      write index to rxbuf_idx_out
    //      write saved header to eth_hdr_out
    //      invalidate descriptor in the data store
    //      instruct status parser to forward accumulated statuses on rxbuf_dma_sts
    //else
    //  {address, max_len} <= rxbuf dma cmd
    //  we emit a DMA command to that address, for the length
    //  we store address+length in the data store
    //  we read one word from data_in, extract header and store it in the data store
    //  we copy length bytes from data_in to data_out

    //separately, we need to monitor statuses from the dma; we do this in a separate process (dataflow function)
    //on each fragment command we send to the datamover, we also send an command, including the buffer id to the status monitoring process,
    //which waits on the corresponding status from the datamover and logs the status in the appropriate storage corresponding
    //to the buffer id; on receipt of a flush status notification, the function will, on receipt of the next
    //corresponding status, flush the accumulated status on rxbuf_dma_sts and clear it in storage

#ifdef ACCL_SYNTHESIS
    hls::stream<rxbuf_status_control> status_instruction;
    #pragma HLS STREAM variable=status_instruction depth=32
#else
    hlslib::Stream<rxbuf_status_control, 32> status_instruction;
#endif

    rxbuf_session_command(
        rxbuf_dma_cmd, 
        rxbuf_idx_in, 
        rxbuf_idx_out, 
        fragment_dma_cmd, 
        session_notification, 
        eth_hdr_in,
        eth_hdr_out, 
        status_instruction
    );
    rxbuf_session_status(status_instruction, fragment_dma_sts, rxbuf_dma_sts);
}