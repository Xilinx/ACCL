#pragma once

#include "ccl_offload_control.h"
#include "eth_intf.h"

typedef struct {
    unsigned int tag;
    unsigned int len;
    unsigned int src;
    unsigned int seqn;
} rxbuf_signature;

typedef struct {
    ap_uint<32> index;
    rxbuf_signature signature;
} rxbuf_notification;

typedef struct {
    ap_uint<64> addr;
    ap_uint<32> index;
    ap_uint<32> len;
    bool valid;
} rxbuf_seek_result;

typedef struct {
    ap_uint<16> index;
    bool first;
    bool last;
} rxbuf_status_control;

typedef struct {
    bool active;
    unsigned int buf_index;
    ap_uint<DEST_WIDTH> mem_index;
    ap_uint<64> address;
    unsigned int remaining;
    eth_header header;
} rxbuf_session_descriptor;

void rxbuf_enqueue(
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &dma_cmd,
    STREAM<ap_uint<32> > &inflight_queue,
    unsigned int *rx_buffers
);

void rxbuf_dequeue(
	STREAM<ap_uint<32> > &dma_sts,
	STREAM<eth_header> &eth_hdr,
	STREAM<ap_uint<32> > &inflight_queue,
	STREAM<rxbuf_notification> &notification_queue,
	unsigned int *rx_buffers
);

void rxbuf_seek(
    STREAM<rxbuf_notification> &rx_notify,
    STREAM<rxbuf_signature> &rx_seek_request,
    STREAM<rxbuf_seek_result> &rx_seek_ack,
    STREAM<ap_uint<32> > &rx_release_request,
	unsigned int *rx_buffers
);

void rxbuf_session(
	STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &rxbuf_dma_cmd, //incoming command for a full DMA transfer targeting a RX buffer
    STREAM<ap_uint<32> > &rxbuf_dma_sts, //status of a rxbuf write (assembled from statuses of fragments)
	STREAM<ap_uint<32> > &rxbuf_idx_in, //indicates RX buffer index corresponding to DMA command on rxbuf_dma_cmd
    STREAM<ap_uint<32> > &rxbuf_idx_out, //forward index of completed RX buffer
    STREAM<ap_axiu<104,0,0,DEST_WIDTH> > &fragment_dma_cmd, //outgoing DMA command for a partial message write to a RX buffer
	STREAM<ap_uint<32> > &fragment_dma_sts, //status of a fragment write
    STREAM<eth_notification> &session_notification, //get notified when there is data for a session
    STREAM<eth_header> &eth_hdr_in, //input header of from depacketizer
    STREAM<eth_header> &eth_hdr_out //forward header of message in completed RX buffer
);
