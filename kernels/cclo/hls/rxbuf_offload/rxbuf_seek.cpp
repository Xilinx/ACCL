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

void rxbuf_seek(
    STREAM<rxbuf_notification> &rx_notify,
    STREAM<rxbuf_signature> &rx_seek_request,
    STREAM<rxbuf_seek_result> &rx_seek_ack,
    STREAM<ap_uint<32> > &rx_release_request,
	unsigned int *rx_buffers
){
#pragma HLS INTERFACE axis port=rx_notify
#pragma HLS INTERFACE axis port=rx_seek_request
#pragma HLS INTERFACE axis port=rx_seek_ack
#pragma HLS INTERFACE axis port=rx_release_request
#pragma HLS INTERFACE m_axi port=rx_buffers	offset=slave bundle=mem
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=1

#ifdef ACCL_SYNTHESIS
    static hls::stream<rxbuf_notification> rx_pending;
    #pragma HLS STREAM variable=rx_pending depth=512
#else
    static hlslib::Stream<rxbuf_notification, 512> rx_pending;
#endif

    static unsigned int num_pending = 0;
    rxbuf_notification pending_notif;
    rxbuf_signature seek_sig;
    rxbuf_seek_result seek_res;
    //if notification, add buffer to queue
    if(!STREAM_IS_EMPTY(rx_notify) && num_pending<512){
        STREAM_WRITE(rx_pending, STREAM_READ(rx_notify));
        num_pending++;
    }
    //if seek request, seek iteratively
    //TODO: this should really be a key-value store but no suitable IP exists
    if(!STREAM_IS_EMPTY(rx_seek_request)){
        seek_res.valid = false;
        seek_sig = STREAM_READ(rx_seek_request);
        for(int i=0; i<num_pending; i++){
            pending_notif = STREAM_READ(rx_pending);
            if((pending_notif.signature.tag == seek_sig.tag || pending_notif.signature.tag == TAG_ANY) && 
                    pending_notif.signature.src == seek_sig.src && pending_notif.signature.seqn == seek_sig.seqn){
                seek_res.addr(31,0) = rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + pending_notif.index * SPARE_BUFFER_FIELDS + ADDRL_OFFSET];
                seek_res.addr(63,32) = rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + pending_notif.index * SPARE_BUFFER_FIELDS + ADDRH_OFFSET];
                seek_res.len = pending_notif.signature.len;
                seek_res.index = pending_notif.index;
                seek_res.valid = true;
                num_pending--;
                break;
            } else{
                STREAM_WRITE(rx_pending, pending_notif);
            }
        }
        STREAM_WRITE(rx_seek_ack, seek_res);
    }
    //if release request, update status of selected buffer
    unsigned int spare_idx;
    if(!STREAM_IS_EMPTY(rx_release_request)){
        spare_idx = STREAM_READ(rx_release_request);
        rx_buffers[(RX_BUFFER_METADATA_OFFSET/4) + spare_idx * SPARE_BUFFER_FIELDS + STATUS_OFFSET] = STATUS_IDLE;
    }
}