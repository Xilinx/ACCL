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

#include "xparameters.h"
#include "ccl_offload_control.h"
#include "mb_interface.h"
#include "microblaze_interrupts_i.h"
#include <setjmp.h>
#define MAX_DMA_TAGS 16

static volatile int 		 use_tcp = 1;
static jmp_buf 				 excp_handler;
static volatile int 		 dma_tag;
static volatile unsigned int next_rx_tag 	 = 0;
static volatile unsigned int num_rx_enqueued = 0;
static volatile unsigned int timeout 	 	 = 1 << 28;
static volatile unsigned int dma_tag_lookup [MAX_DMA_TAGS]; //index of the spare buffer that has been issued with that dma tag. -1 otherwise
static volatile	unsigned int dma_transaction_size 	= DMA_MAX_BTT;
static volatile unsigned int max_dma_in_flight 		= DMA_MAX_TRANSACTIONS;

static datapath_arith_config arcfg;
static communicator world;

unsigned int enqueue_rx_buffers(void);
int  dequeue_rx_buffers(void);

//circular buffer operation
void cb_init(circular_buffer *cb, unsigned int capacity){
    cb->write_idx=0;
    cb->read_idx=0;
    cb->occupancy=0;
    cb->capacity= (capacity < MAX_CIRCULAR_BUFFER_SIZE) ? capacity : MAX_CIRCULAR_BUFFER_SIZE;
}

void cb_push(circular_buffer *cb, unsigned int item){
    cb->buffer[cb->write_idx] = item;
    cb->write_idx = (cb->write_idx + 1) % cb->capacity;
    cb->occupancy++;
}

unsigned int cb_pop(circular_buffer *cb){
    unsigned int item = cb->buffer[cb->read_idx];
    cb->read_idx = (cb->read_idx + 1) % cb->capacity;
    cb->occupancy--;
    return item;
}

//IRQ CTRL

//enable interrupts from interrupt controller
static inline void enable_irq(){
    //set master enable register  MER
    Xil_Out32(IRQCTRL_BASEADDR + IRQCTRL_MASTER_ENABLE_REGISTER_OFFSET, IRQCTRL_MASTER_ENABLE_REGISTER_HARDWARE_INTERRUPT_ENABLE|IRQCTRL_MASTER_ENABLE_REGISTER_MASTER_ENABLE);
}
//disable interrupts from interrupt controller
static inline void disable_irq(){
    //set interrupt enable register IER with correct mask
    //unset master enable register  MER
    Xil_Out32(IRQCTRL_BASEADDR + IRQCTRL_MASTER_ENABLE_REGISTER_OFFSET, 0);
}

// set interrupt controller
// mask:
// IRQCTRL_TIMER_ENABLE              0x0001
// IRQCTRL_DMA0_CMD_QUEUE_EMPTY      0x0002
// IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY  0x0004
static inline void set_irq(int mask){
    //set interrupt enable register IER with correct mask
    Xil_Out32(IRQCTRL_BASEADDR + IRQCTRL_INTERRUPT_ENABLE_REGISTER_OFFSET, mask);
}

// set interrupt controller
// mask:
// IRQCTRL_TIMER_ENABLE              0x0001
// IRQCTRL_DMA0_CMD_QUEUE_EMPTY      0x0002
// IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY  0x0004
static inline void setup_irq(int mask){
    disable_irq();
    //set INTERRUPT_ENABLE_REGISTER IER with correct mask
    set_irq(mask);
    enable_irq();
}

// acknowledge irq complete
// mask:
// IRQCTRL_TIMER_ENABLE              0x0001
// IRQCTRL_DMA0_CMD_QUEUE_EMPTY      0x0002
// IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY  0x0004
static inline void ack_irq(int mask){
    //set INTERRUPT_ACKNOWLEDGE_REGISTER IER with the mask
    Xil_Out32(IRQCTRL_BASEADDR + IRQCTRL_INTERRUPT_ACKNOWLEDGE_REGISTER_OFFSET, mask);
}

//gets INTERRUPT_PENDING_REGISTER 
// that holds a 1 in bit i-esim means a that interrupt i-esim occurred  
static inline int get_irq(void){
    return Xil_In32(IRQCTRL_BASEADDR + IRQCTRL_INTERRUPT_PENDING_REGISTER_OFFSET);
}

//AXI Timer

//creates a timeout that will interrupt the MB code execution 
static inline void start_timeout(unsigned int time){
    //1. set timer load register x(TLRx) with time so that TIMING_INTERVAL = (TLRx + 2) * AXI_CLOCK_PERIOD
    //2. load timer register
    //3. set counter enable/down/non-autoreload/interrupt enable/clear load /generate mode
    Xil_Out32(TIMER_BASEADDR + TIMER0_LOAD_REGISTER_OFFSET, time);
    Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_LOAD_TIMER_MASK);
    Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_ENABLE_MASK | CONTROL_AND_STATUS_REGISTER_INTERRUPT_ENABLE_MASK | CONTROL_AND_STATUS_REGISTER_UP_DOWN_MASK);
}

//cancel the timeout
static inline void cancel_timeout(){
    //1. set counter disable/interrupt disable
    Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_INTERRUPT_MASK);
}

//clear timer interrupt
static inline void clear_timer_interrupt(){
    Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_INTERRUPT_MASK);
}

//creates a timeout that will interrupt the MB code execution 
static inline void start_timer1(){
    unsigned int init_time = 0;
    //1. set timer load register x(TLRx) with 0 
    //2. load timer register
    //3. set counter enable/down/non-autoreload/interrupt enable/clear load /generate mode
    Xil_Out32(TIMER_BASEADDR + TIMER1_LOAD_REGISTER_OFFSET, init_time);
    Xil_Out32(TIMER_BASEADDR + TIMER1_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_LOAD_TIMER_MASK);
    Xil_Out32(TIMER_BASEADDR + TIMER1_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_ENABLE_MASK );
}

static inline unsigned int read_timer1(){
    return Xil_In32(TIMER_BASEADDR + TIMER1_COUNTER_REGISTER_OFFSET);
}

void stream_isr(void) {
    int n_enqueued;
    int irq_mask;
    int irq = get_irq();
    if (irq & IRQCTRL_TIMER_ENABLE){
        clear_timer_interrupt();
    }
    if (irq & (use_tcp ?  IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY : IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY) ){
        dequeue_rx_buffers();
    } 

    n_enqueued = enqueue_rx_buffers();
    
    if ( n_enqueued == 0 && ((irq & (use_tcp ? IRQCTRL_DMA2_CMD_QUEUE_EMPTY : IRQCTRL_DMA0_CMD_QUEUE_EMPTY) ) || (irq & IRQCTRL_TIMER_ENABLE)) )
    {   //if no spare buffer is present in the cmd queue disable irq_from empty CMD queue as it could lead to starvation
        irq_mask = (use_tcp ? IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY: IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY) | IRQCTRL_TIMER_ENABLE;
        setup_irq(irq_mask);
        start_timeout(100000);
    }else{
        irq_mask = (use_tcp ? IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY: IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY) | (use_tcp ? IRQCTRL_DMA2_CMD_QUEUE_EMPTY: IRQCTRL_DMA0_CMD_QUEUE_EMPTY);
        setup_irq(irq_mask);
        cancel_timeout();
    }
    ack_irq(irq);
}
//DMA functions

//issue a dma command 
static inline void dma_cmd_addrh_addrl(unsigned int channel, unsigned int btt, unsigned int addrh, unsigned int addrl, unsigned int tag){
    putd(channel, 0xC0800000 | btt); // 31=DRR 30=EOF 29-24=DSA 23=Type 22-0=BTT
    putd(channel, addrl);
    putd(channel, addrh);
    cputd(channel, 0x2000 | tag); 	 // 15-12=xCACHE 11-8=xUSER 7-4=RSVD 3-0=TAG
}

//start a DMA operations on a specific channel
static inline void dma_cmd(unsigned int channel, unsigned int btt, uint64_t addr, unsigned int tag) {
    unsigned int addrl = addr & 0xFFFFFFFF;
    unsigned int addrh = (addr >> 32) & 0xFFFFFFFF;

    dma_cmd_addrh_addrl( channel,  btt, addrh, addrl, tag);
}

static inline void dma_cmd_without_EOF(unsigned int channel, unsigned int btt, uint64_t addr, unsigned int tag) {
    unsigned int addrl = addr & 0xFFFFFFFF;
    unsigned int addrh = (addr >> 32) & 0xFFFFFFFF;

    putd(channel, 0x80800000 | btt); // 31=DRR 30=EOF 29-24=DSA 23=Type 22-0=BTT
    putd(channel, addrl);
    putd(channel, addrh);
    cputd(channel, 0x2000 | tag); 	 // 15-12=xCACHE 11-8=xUSER 7-4=RSVD 3-0=TAG
}

//check DMA status and in case recalls the exception handler
void check_DMA_status(unsigned int status, unsigned int expected_btt, unsigned int tag, unsigned int indeterminate_btt_mode_active){
    // 3-0 TAG 
    // 4 INTERNAL 	ERROR usually a btt=0 trigger this
    // 5 DECODE 	ERROR address decode error timeout
    // 6 SLAVE 		ERROR DMA encountered a slave reported error
    // 7 OKAY		the associated transfer command has been completed with the OKAY response on all intermediate transfers.	 
    if ((status & 0x000f) != tag ){
        longjmp(excp_handler, DMA_MISMATCH_ERROR);
    }
    if ( status & 0x0010){
        longjmp(excp_handler, DMA_INTERNAL_ERROR);
    }
    if ( status & 0x0020){
        longjmp(excp_handler, DMA_DECODE_ERROR);
    }
    if ( status & 0x0040){
        longjmp(excp_handler, DMA_SLAVE_ERROR);
    }
    if ( !(status & 0x0080)){
        longjmp(excp_handler, DMA_NOT_OKAY_ERROR);
    }
    if(indeterminate_btt_mode_active==1){
        //30-8 		BYTES received
        //31 		END OF PACKET indicates that S2MM received a TLAST 
        if ( !(status & 0x80000000)){
            longjmp(excp_handler, DMA_NOT_END_OF_PACKET_ERROR);
        }

        if ( ( (status & 0x7FFFFF00) >> 8) != expected_btt ){
            longjmp(excp_handler, DMA_NOT_EXPECTED_BTT_ERROR);
        }
    }
}

//AXIS SWITCH management

//configure a acis switch path from a source to a destination
//PG085 page 26
//config registers start at 0x40
static inline void set_switch_datapath(unsigned int base, unsigned int src, unsigned int dst) {
    Xil_Out32(base+0x40+4*dst, src);
}

//disable a axis switch master output port
//PG085 page 26
//config registers start at 0x40
static inline void disable_switch_datapath(unsigned int base, unsigned int dst) {
    Xil_Out32(base+0x40+4*dst, 0x80000000);
}
//procedure to apply configurations of axis switch
//PG085 page 26
static inline void apply_switch_config(unsigned int base) {
    //update registers
    Xil_Out32(base, 2);
    //wait for the switch to come back online and clear control register
    unsigned int count = 0;
    while(Xil_In32(base) != 0){
        if(timeout != 0 && count >= timeout )
            longjmp(excp_handler, CONFIG_SWITCH_ERROR);
        count ++;
    }
}

//configure the switches for various scenarios:
//1 == DATAPATH_DMA_LOOPBACK:   DMA1 RX -> EXT_KRNL_TX
 //                             EXT_KRNL_RX -> DMA1 TX
//
//2 == DATAPATH_DMA_REDUCTION:  DMA0 RX -> Arith Op0
//   							DMA1 RX -> Arith Op1 
//   							Arith Res -> EXT_KRNL_TX
//                              EXT_KRNL_RX -> DMA1 TX 
//
//3 == DATAPATH_OFFCHIP_TX_UDP: DMA1 RX -> EXT_KRNL_TX
//                              EXT_KRNL_RX -> UDP TX
//
//4 == DATAPATH_OFFCHIP_TX_TCP: DMA1 RX -> EXT_KRNL_TX
//                              EXT_KRNL_RX -> TCP TX
//
//5 == DATAPATH_OFFCHIP_UDP_REDUCTION: 	DMA0 RX -> Arith Op0
//   									DMA1 RX -> Arith Op1 
//   							        Arith Res -> EXT_KRNL_TX
//                                      EXT_KRNL_RX -> UDP TX 
//
//6 == DATAPATH_OFFCHIP_TCP_REDUCTION: 	DMA0 RX -> Arith Op0
//   									DMA1 RX -> Arith Op1 
//   							        Arith Res -> EXT_KRNL_TX
//                                      EXT_KRNL_RX -> TCP TX 
void configure_switch(unsigned int scenario) {
    switch (scenario)
    {
        case DATAPATH_DMA_LOOPBACK:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_EXT_KRNL);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_EXT_KRNL, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP0);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP1);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_UDP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_TCP_TX);			
            break;
        case DATAPATH_DMA_REDUCTION:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA0_RX, SWITCH_M_ARITH_OP0);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_ARITH_OP1);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_ARITH_RES, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_UDP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_TCP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_EXT_KRNL);	
            break;
        case DATAPATH_OFFCHIP_TX_UDP:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_EXT_KRNL);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_EXT_KRNL, SWITCH_M_UDP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP0);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP1);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_TCP_TX);	
            break;
        case DATAPATH_OFFCHIP_TX_TCP:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_EXT_KRNL);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_EXT_KRNL, SWITCH_M_TCP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP0);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_ARITH_OP1);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_UDP_TX);	
            break;
        case DATAPATH_OFFCHIP_UDP_REDUCTION:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA0_RX, SWITCH_M_ARITH_OP0);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_ARITH_OP1);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_ARITH_RES, SWITCH_M_EXT_KRNL);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_EXT_KRNL, SWITCH_M_UDP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_TCP_TX);	
            break;
        case DATAPATH_OFFCHIP_TCP_REDUCTION:
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA0_RX, SWITCH_M_ARITH_OP0);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_DMA1_RX, SWITCH_M_ARITH_OP1);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_ARITH_RES, SWITCH_M_EXT_KRNL);
            set_switch_datapath(SWITCH_BASEADDR, SWITCH_S_EXT_KRNL, SWITCH_M_TCP_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_DMA1_TX);
            disable_switch_datapath(SWITCH_BASEADDR, SWITCH_M_UDP_TX);
            break;
        default:
            return;
    }
    apply_switch_config(SWITCH_BASEADDR);
}

//Configure external parts of the datapath
//by setting appropriate TDEST on streams
static inline void configure_plugins(unsigned int acfg, unsigned int d0cfg, unsigned int d1cfg, unsigned int c0cfg){
    unsigned int val = 0;
    val = (c0cfg<<24) | (d1cfg<<16) | (d0cfg<<8) | acfg;
    Xil_Out32(GPIO_TDEST_BASEADDR, val);
}

//Configure entire datapath (internal switch and plugins)
//from operand and result types (compressed/uncompressed) and compression config
void configure_datapath(unsigned int scenario,
                        unsigned int function,
                        unsigned int compression){
    configure_switch(scenario);
    int cfg_arith, cfg_clane_op0, cfg_clane_op1, cfg_clane_res;
    //find and set arithmetic function config
    if(function >= MAX_REDUCE_FUNCTIONS){
        longjmp(excp_handler, COMPRESSION_ERROR);
    }
    cfg_arith = arcfg.arith_tdest[function];
    //find and set configs for each compression lane
    if( (scenario == DATAPATH_DMA_REDUCTION) || 
            (scenario == DATAPATH_OFFCHIP_UDP_REDUCTION) || 
                (scenario == DATAPATH_OFFCHIP_TCP_REDUCTION)    ){
        //we're doing a reduction: op0, op1 -> res
        if(arcfg.arith_is_compressed && (){
            cfg_clane_op0 = (compression & OP0_COMPRESSED) ? NO_COMPRESSION : arcfg.compressor_tdest;
            cfg_clane_op1 = (compression & OP1_COMPRESSED) ? NO_COMPRESSION : arcfg.compressor_tdest;
            cfg_clane_res = (compression & RES_COMPRESSED) ? NO_COMPRESSION : arcfg.compressor_tdest;
        } else {
            cfg_clane_op0 = (compression & OP0_COMPRESSED) ? arcfg.decompressor_tdest : NO_COMPRESSION;
            cfg_clane_op1 = (compression & OP1_COMPRESSED) ? arcfg.decompressor_tdest : NO_COMPRESSION;
            cfg_clane_res = (compression & RES_COMPRESSED) ? arcfg.decompressor_tdest : NO_COMPRESSION;
        }
    } else {
        //we're not doing a copy or send: op0 -> res
        cfg_clane_op0 = NO_COMPRESSION;
        cfg_clane_op1 = NO_COMPRESSION;
        if(compression & OP0_COMPRESSED){
            cfg_clane_res = (compression & RES_COMPRESSED) ? NO_COMPRESSION : arcfg.decompressor_tdest;
        } else {
            cfg_clane_res = (compression & RES_COMPRESSED) ? arcfg.compressor_tdest : NO_COMPRESSION;
        } 
    }
    configure_plugins(cfg_arith, cfg_clane_op0, cfg_clane_op1, cfg_clane_res);
}

//retrieves the communicator
static inline communicator find_comm(unsigned int adr){
    communicator ret;
    ret.size 		= Xil_In32(adr);
    ret.local_rank 	= Xil_In32(adr+4);
    if(ret.size != 0 && ret.local_rank < ret.size){
        ret.ranks = (comm_rank*)(adr+8);
    } else {
        ret.size = 0;
        ret.local_rank = 0;
        ret.ranks = NULL;
    }
    return ret;
}

//Packetizer/Depacketizer
static inline void start_packetizer(unsigned int base_addr,unsigned int max_pktsize) {
    //get number of 512b/64B transfers corresponding to max_pktsize
    unsigned int max_pkt_transfers = (max_pktsize+63)/64;
    Xil_Out32(base_addr+0x10, max_pkt_transfers);
    SET(base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
}

static inline void start_depacketizer(unsigned int base_addr) {
    SET(base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK );
}

//create the instructions for packetizer to send a message. Those infos include, among the others the message header.
unsigned int start_packetizer_message(unsigned int dst_rank, unsigned int len, unsigned int tag){
    unsigned int src_rank, sequence_number, dst, net_tx, curr_len, i;
    unsigned int transmitted = 0; 
    //prepare header arguments
    
    src_rank 		= world.local_rank;
    sequence_number = world.ranks[dst_rank].outbound_seq;
    if (use_tcp){
        dst 			= world.ranks[dst_rank].session; 
        net_tx 			= CMD_TCP_TX; 
    }else{
        dst		 		= world.ranks[dst_rank].port; 
        net_tx		 	= CMD_UDP_TX; 
    }
    //enqueue message headers
    for(i=0; len > 0 && i < max_dma_in_flight; i++){
        sequence_number++;

        if(len > dma_transaction_size)
            curr_len = dma_transaction_size;
        else
            curr_len = len;

        putd(net_tx, dst); 
        putd(net_tx, curr_len);
        putd(net_tx, tag);
        putd(net_tx, src_rank);
        cputd(net_tx,sequence_number);
        len 		-= curr_len;
        transmitted += curr_len;
    }
    return i;
}

//check that packetizer has finished processing every packet it was supposed to.
static inline void ack_packetizer(unsigned int dst_rank,unsigned int num_msg){
    unsigned int count, ack_seq_num;
    unsigned int packetizer_sts_stream = use_tcp ? STS_TCP_PKT : STS_UDP_PKT;
    while(num_msg > 0){

        for(count=0; tngetd(packetizer_sts_stream); count++){
            if(count > timeout){
                longjmp(excp_handler, PACK_TIMEOUT_STS_ERROR);
            }
        }
        ack_seq_num = getd(packetizer_sts_stream);
        if(	world.ranks[dst_rank].outbound_seq+1 == ack_seq_num){
            world.ranks[dst_rank].outbound_seq+=1;
        } else{
            longjmp(excp_handler, PACK_SEQ_NUMBER_ERROR);
        }
        num_msg -=1;
    }
}

//TCP connection management

//establish connection with every other rank in the communicator
int openCon()
{
    unsigned int session 	= 0;
    unsigned int dst_ip 	= 0;
    unsigned int dst_port 	= 0;
    unsigned int success	= 0;
    int ret = 0;

    unsigned int cur_rank_ip	= 0;
    unsigned int cur_rank_port 	= 0;

    unsigned int size 		= world.size;
    unsigned int local_rank = world.local_rank;

    //open connection to all the ranks except for the local rank
    for (int i = 0; i < size; i++)
    {
        if (i != local_rank)
        {
            cur_rank_ip 	= world.ranks[i].ip;
            cur_rank_port 	= world.ranks[i].port;
            //send open connection request to the packetizer
            putd(CMD_TCP_CON, cur_rank_ip);
            putd(CMD_TCP_CON, cur_rank_port);
        }
    }
    ret = COLLECTIVE_OP_SUCCESS;
    //wait until the connections status are all returned
    for (int i = 0; i < size -1 ; i++)
    {	
        //ask: tngetd or the stack will return a non-success? 
        session 	= getd(STS_TCP_CON);
        dst_ip 		= getd(STS_TCP_CON);
        dst_port 	= getd(STS_TCP_CON);
        success 	= getd(STS_TCP_CON);

        if (success)
        {
            //store the session ID into corresponding rank
            for (int j = 0; j < size ; ++j)
            {
                cur_rank_ip 	= world.ranks[j].ip;
                cur_rank_port 	= world.ranks[j].port;
                if ((dst_ip == cur_rank_ip) && (dst_port == cur_rank_port))
                {
                    world.ranks[j].session = session;
                    break;
                }
            }
        }
        else
        {
            ret = OPEN_COM_NOT_SUCCEEDED;
        }
    }
    return ret;
}
//open local port for listening
int openPort()
{
    int success = 0;

    //open port with only the local rank
    putd(CMD_TCP_PORT, world.ranks[world.local_rank].port);
    success = getd(STS_TCP_PORT);

    if (success)
        return COLLECTIVE_OP_SUCCESS;
    else
        return OPEN_PORT_NOT_SUCCEEDED;

}
//function that was supposed to openPort AND to openCon. 
//Since at this level we are not sure the other ccl_offlaod instances have open their port 
//this function is not working.
int init_connection()
{
    int ret_openPort 	= openPort();
    int ret_openCon 	= openCon();

    int retval = (ret_openPort | ret_openCon);

    return retval;
}

static inline unsigned int segment(unsigned int number_of_bytes,unsigned int segment_size){
    return  (number_of_bytes + segment_size - 1) / segment_size;
} 

//DMA0/2 TX management 

//enques cmd from DMA that receives from network stack. 
//RX address queue management
//maintaint a list of N buffers in device memory
//queue those buffers for receives
unsigned int enqueue_rx_buffers(void){
    unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
    unsigned int ret = 0, cmd_queue;
    int i,new_dma_tag;
    cmd_queue = use_tcp ? CMD_DMA2_TX : CMD_DMA0_TX;
    for(i=0; i<nbufs; i++){
        //if((rx_buf_list[i].enqueued == 1) && (rx_buf_list[i].dma_tag == next_rx_tag)) return;
        if(num_rx_enqueued >= 16) return ret;
        if(rx_buf_list[i].status   != STATUS_IDLE) continue;
        //found a spare buffer to enqueue 
        //look for a new dma tag
        for(new_dma_tag=0; new_dma_tag < MAX_DMA_TAGS && dma_tag_lookup[new_dma_tag] != -1; new_dma_tag++);
        //new_dma_tag now holds the new dma tag to use
        if( new_dma_tag >= MAX_DMA_TAGS) return ret; //but something probably wrong in num_rx_enqueued
    
        //whatever we find now we can enqueue
        dma_cmd_addrh_addrl(cmd_queue, DMA_MAX_BTT, rx_buf_list[i].addrh, rx_buf_list[i].addrl, new_dma_tag);
        rx_buf_list[i].status 	= STATUS_ENQUEUED;
        rx_buf_list[i].dma_tag 	= new_dma_tag;
        dma_tag_lookup[new_dma_tag] = i;
        //next_rx_tag = (next_rx_tag + 1) & 0xf;
        num_rx_enqueued++;
        ret ++;
    }

    return ret;
}

//dequeue sts from DMA that receives from network stack. 
int dequeue_rx_buffers(void){
    //test if rx channel is non-empty and if so, read it
    unsigned int invalid, status, dma_tag, dma_tx_id, net_rx_id,count = 0;
    int spare_buffer_idx;
    unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
    if(use_tcp){
        dma_tx_id = STS_DMA2_TX;
        net_rx_id = STS_TCP_RX;
    }else{
        dma_tx_id = STS_DMA0_TX;
        net_rx_id = STS_UDP_RX;
    }

    do {
        if(timeout != 0 && count >= timeout )
            longjmp(excp_handler, DEQUEUE_BUFFER_TIMEOUT_ERROR);
        count ++;
        invalid = tngetd(dma_tx_id); 
        if(invalid == 0){
            count = 0;
            status = getd(dma_tx_id);
            dma_tag = status & 0xf;
            spare_buffer_idx = dma_tag_lookup[dma_tag];
            
            if (rx_buf_list[spare_buffer_idx].status != STATUS_ENQUEUED){
                longjmp(excp_handler, DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR);
            }
            if( rx_buf_list[spare_buffer_idx].dma_tag != dma_tag ){
                longjmp(excp_handler, DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH);
            } 
            if(spare_buffer_idx >= nbufs){
                longjmp(excp_handler, DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR);
            }
            rx_buf_list[spare_buffer_idx].rx_len 		  	= getd(net_rx_id);
            rx_buf_list[spare_buffer_idx].rx_tag 		  	= getd(net_rx_id);
            rx_buf_list[spare_buffer_idx].rx_src 		  	= getd(net_rx_id);
            rx_buf_list[spare_buffer_idx].sequence_number 	= getd(net_rx_id);
            rx_buf_list[spare_buffer_idx].status 			= STATUS_RESERVED;
            check_DMA_status(status, rx_buf_list[spare_buffer_idx].rx_len, rx_buf_list[spare_buffer_idx].dma_tag, 1);
            num_rx_enqueued--;
            dma_tag_lookup[dma_tag] = -1;
            
        }
    } while (invalid == 0);
    return 0;
}

//basic functionalities: send/recv/copy/accumulate
static inline int start_dma( 	
                    unsigned int len,
                    uint64_t op0_addr,
                    uint64_t op1_addr,
                    uint64_t res_addr,
                    unsigned int which_dma,
                    unsigned int casting,
                    unsigned int dma_tag_tmp) {
    //BE CAREFUL!: if(len > DMA_MAX_BTT) return DMA_TRANSACTION_SIZE;
    //NOTE: len is the number of uncompressed elements to be transferred

    //set up len in the context of casting
    unsigned int byte_len;
    unsigned int num_compressed_elems = (len+arcfg.elem_ratio-1)/arcfg.elem_ratio;
    unsigned int compressed_byte_len = (num_compressed_elems*arcfg.compressed_elem_bits+7)/8;
    unsigned int uncompressed_byte_len = (len*arcfg.uncompressed_elem_bits+7)/8;

    //start DMAs 
    if( which_dma & USE_OP0_DMA){
        dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
        byte_len = (casting & OP0_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        dma_cmd(CMD_DMA0_RX, byte_len, op0_addr, dma_tag_tmp);
    }
    if( which_dma & USE_OP1_DMA){
        dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
        byte_len = (casting & OP1_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        dma_cmd(CMD_DMA1_RX, byte_len, op1_addr, dma_tag_tmp);
    }
    if( which_dma & USE_RES_DMA){
        dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
        byte_len = (casting & RES_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        dma_cmd(CMD_DMA1_TX, byte_len, res_addr, dma_tag_tmp);
    }
    if( which_dma & USE_RES_DMA_WITHOUT_TLAST){
        dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
        byte_len = (casting & RES_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        dma_cmd_without_EOF(CMD_DMA1_TX, byte_len, res_addr, dma_tag_tmp);
    }
    return dma_tag_tmp;
}

static inline void ack_dma(
    unsigned int len,
    unsigned int which_dma,
    unsigned int casting) {
    unsigned int invalid, count, status;

    //NOTE: len is the number of uncompressed elements to be transferred

    //set up len in the context of casting
    unsigned int byte_len;
    unsigned int num_compressed_elems = (len+arcfg.elem_ratio-1)/arcfg.elem_ratio;
    unsigned int compressed_byte_len = (num_compressed_elems*arcfg.compressed_elem_bits+7)/8;
    unsigned int uncompressed_byte_len = (len*arcfg.uncompressed_elem_bits+7)/8;

    //wait for DMAs to complete
    count = 0;
    do {
        if(timeout != 0 && count >= timeout )
            longjmp(excp_handler, DMA_TIMEOUT_ERROR);
        count ++;
        invalid = 0;
        if( which_dma & USE_OP0_DMA){
            invalid += tngetd(STS_DMA0_RX);
        }
        if( which_dma & USE_OP1_DMA){
            invalid += tngetd(STS_DMA1_RX);
        }
        if( which_dma & (USE_RES_DMA | USE_RES_DMA_WITHOUT_TLAST)){
            invalid += tngetd(STS_DMA1_TX);
        }
    } while (invalid);

    if( which_dma & USE_OP0_DMA){
        status = getd(STS_DMA0_RX);
        dma_tag = (dma_tag + 1) & 0xf;
        byte_len = (casting & OP0_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        check_DMA_status(status, byte_len, dma_tag, 0);
    }
    if( which_dma & USE_OP1_DMA){
        status = getd(STS_DMA1_RX);
        dma_tag = (dma_tag + 1) & 0xf;
        byte_len = (casting & OP1_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        check_DMA_status(status, byte_len, dma_tag, 0);
    }
    if( which_dma & USE_RES_DMA){
        status = getd(STS_DMA1_TX);
        dma_tag = (dma_tag + 1) & 0xf;
        byte_len = (casting & RES_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        check_DMA_status(status, byte_len, dma_tag, 1);
    }
    if( which_dma & USE_RES_DMA_WITHOUT_TLAST){
        status = getd(STS_DMA1_TX);
        dma_tag = (dma_tag + 1) & 0xf;
        byte_len = (casting & RES_COMPRESSED) ? compressed_byte_len : uncompressed_byte_len;
        check_DMA_status(status, byte_len, dma_tag, 0);
    }
}

//configure datapath before calling this method
//segment a logical dma command in multiple dma commands.
//if one of the DMA is not used fill the corresponding address with NULL
int move_data(
    unsigned int len,
    uint64_t op0_addr,
    uint64_t op1_addr,
    uint64_t res_addr,
    unsigned int which_blocks,
    unsigned int casting,
    unsigned dst_rank,
    unsigned mpi_tag
) {
    unsigned int remaining_to_move, remaining_to_ack, dma_tag_tmp, curr_len_move, curr_len_ack, i;
    remaining_to_move 	= len;
    remaining_to_ack 	= len;
    dma_tag_tmp 		= dma_tag;

    //default size of transaction is full size
    curr_len_move = dma_transaction_size;
    curr_len_ack  = dma_transaction_size;
    //1. issue at most max_dma_in_flight of dma_transaction_size
    //start packetizer
    if(which_blocks & USE_PACKETIZER){
        start_packetizer_message(dst_rank, len, mpi_tag);
    }
    for (i = 0; remaining_to_move > 0 && i < max_dma_in_flight ; i++){
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, op0_addr, op1_addr, res_addr, which_blocks, casting, dma_tag_tmp);
        remaining_to_move 	-= curr_len_move;

        //increment addresses based on casting spec
        if(casting & OP0_COMPRESSED){
            op0_addr 		+= curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            op0_addr 		+= curr_len_move;
        }
        if(casting & OP1_COMPRESSED){
            op1_addr 		+= curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            op1_addr 		+= curr_len_move;
        }
        if(casting & RES_COMPRESSED){
            res_addr 		+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            res_addr 		+= curr_len_move;
        }
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(remaining_to_move > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, which_blocks, casting);
        //ack pack
        if(which_blocks & USE_PACKETIZER){
            ack_packetizer(dst_rank, 1);
        }
        remaining_to_ack  -= curr_len_ack;

        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //start pack
        if(which_blocks & USE_PACKETIZER){
            start_packetizer_message(dst_rank, curr_len_move, mpi_tag);
        }
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, op0_addr, op1_addr, res_addr, which_blocks, casting, dma_tag_tmp);
        remaining_to_move 	-= curr_len_move;

        //increment addresses based on casting spec
        if(casting & USE_OP0_DMA){
            op0_addr 		+= curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            op0_addr 		+= curr_len_move;
        }
        if(casting & USE_OP1_DMA){
            op1_addr 		+= curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            op1_addr 		+= curr_len_move;
        }
        if(casting & USE_RES_DMA){
            res_addr 		+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
        } else {
            res_addr 		+= curr_len_move;
        }
    }
    //3. finish ack the remaining
    while(remaining_to_ack > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, which_blocks, casting);
        //ack pack
        if(which_blocks & USE_PACKETIZER){
            ack_packetizer(dst_rank, 1);
        }
        remaining_to_ack  -= curr_len_ack;
    }
    return COLLECTIVE_OP_SUCCESS;
}

//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
//No compression/decompression, don't care arithmetic
static inline int copy(	unsigned int len,
                        uint64_t src_addr,
                        uint64_t dst_addr,
                        unsigned int compression) {
    configure_datapath(DATAPATH_DMA_LOOPBACK, 0, compression);
    return move_data(len, src_addr, NULL, dst_addr, USE_OP0_DMA | USE_RES_DMA, compression, 0, 0);
}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx back to dst buffer
int reduce_loopback(unsigned int len,
                    unsigned int function,
                    uint64_t op0_addr,
                    uint64_t op1_addr,
                    uint64_t res_addr,
                    unsigned int compression) {
    configure_datapath( DATAPATH_DMA_REDUCTION, function, compression);
    return move_data(len, op0_addr, op1_addr, res_addr, USE_OP0_DMA | USE_OP1_DMA | USE_RES_DMA, compression, 0, 0);
}

//transmits a buffer to a rank of the world communicator
//right now we assume the transmit is always from a user source buffer
//so we will apply whatever casting is appropriate for that
//TODO: safer implementation.
int send(
    unsigned int dst_rank,
    unsigned int len,
    uint64_t src_addr,
    unsigned int dst_tag,
    unsigned int compression
){
    configure_datapath(use_tcp ? DATAPATH_OFFCHIP_TX_TCP : DATAPATH_OFFCHIP_TX_UDP, 0, compression);
    return move_data(len, src_addr, NULL, NULL, USE_OP0_DMA | USE_PACKETIZER, compression, dst_rank, dst_tag);
}

//performs a reduction using DMA0 and DMA1 and then sends the result 
//through tx_subsystem to dst_rank
static inline int fused_reduce_send(
                    unsigned int dst_rank,
                    unsigned int len,
                    uint64_t op0_addr,
                    uint64_t op1_addr,
                    unsigned int dst_tag,
                    unsigned int compression) {
    configure_datapath(use_tcp ? DATAPATH_OFFCHIP_TCP_REDUCTION : DATAPATH_OFFCHIP_UDP_REDUCTION, 0, compression);
    return move_data(len, op0_addr, op1_addr, NULL, USE_OP0_DMA | USE_OP1_DMA | USE_PACKETIZER, compression, dst_rank, dst_tag);
}

//iterates over rx buffers until match is found or a timeout expires
//matches len, src and tag if tag is not ANY
//returns the index of the spare_buffer or -1 if not found
int seek_rx_buffer(
                        unsigned int src_rank,
                        unsigned int len,
                        unsigned int src_tag
                    ){
    unsigned int seq_num; //src_port TODO: use this variable to choose depending on tcp/udp session id or port
    int i;
    seq_num = world.ranks[src_rank].inbound_seq + 1;
    //TODO: use a list to store recent message received to avoid scanning in the entire spare_buffer_struct.
    //parse rx buffers until match is found
    //matches len, src
    //matches tag or tag is ANY
    //return buffer index 
    unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
    for(i=0; i<nbufs; i++){	
        microblaze_disable_interrupts();
        if(rx_buf_list[i].status == STATUS_RESERVED)
        {
            if((rx_buf_list[i].rx_src == src_rank) && (rx_buf_list[i].rx_len == len))
            {
                if(((rx_buf_list[i].rx_tag == src_tag) || (src_tag == TAG_ANY)) && (rx_buf_list[i].sequence_number == seq_num) )
                {
                    //only now advance sequence number
                    world.ranks[src_rank].inbound_seq++;
                    microblaze_enable_interrupts();
                    return i;
                }
            }
        }
        microblaze_enable_interrupts();
    }
    return -1;
}

//iterates over rx buffers until match is found or a timeout expires
//matches len, src and tag if tag is not ANY
//returns the index of the spare_buffer
//timeout is jumps to exception handler
int wait_on_rx(	unsigned int src_rank,
                    unsigned int len,
                    unsigned int src_tag){
    unsigned int count;
    int i;
    for(count = 0; timeout == 0 || count < timeout; count ++){
        i = seek_rx_buffer(src_rank, len, src_tag);
        if(i >= 0) return i;
    }
    longjmp(excp_handler, RECEIVE_TIMEOUT_ERROR);
}

//waits for a messages to come and move their contents in a buffer
int recv(	unsigned int src_rank,	
            unsigned int len,
            uint64_t dst_addr,
            unsigned int src_tag,
            unsigned int compression){
    uint64_t rx_buff_addr;
    unsigned int buf_idx;
    unsigned int remaining_to_move, remaining_to_ack, dma_tag_tmp, curr_len_move, curr_len_ack, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    configure_datapath(DATAPATH_DMA_LOOPBACK, 0, compression);
    
    remaining_to_move 	= len;
    remaining_to_ack 	= len;
    dma_tag_tmp 		= dma_tag;

    //default size of transaction is full size
    curr_len_move = dma_transaction_size;
    curr_len_ack  = dma_transaction_size;
    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; remaining_to_move > 0 && i < max_dma_in_flight ; i++){
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp  = start_dma(curr_len_move, rx_buff_addr, 0, dst_addr, USE_OP0_DMA | USE_RES_DMA, USE_RES_DMA, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        dst_addr 			+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(remaining_to_move > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP1_DMA | USE_RES_DMA, USE_RES_DMA);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
        //enqueue other DMA movement
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp = start_dma(curr_len_move, rx_buff_addr, 0, dst_addr, USE_OP0_DMA | USE_RES_DMA, USE_RES_DMA, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        dst_addr 			+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
    }
    //3. finish ack the remaining
    while(remaining_to_ack > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP1_DMA | USE_RES_DMA, USE_RES_DMA);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
    }
    return COLLECTIVE_OP_SUCCESS;
}

//1) receives from a rank 
//2) sums with a a buffer 
//3) the result is saved in (possibly another) local buffer
int fused_recv_reduce(
        unsigned int src_rank,
        unsigned int len,
        unsigned int func,
        uint64_t op0_addr,
        uint64_t dst_addr,
        unsigned int src_tag,
        unsigned int compression)
    {
    uint64_t rx_buff_addr;
    unsigned int buf_idx;
    unsigned int remaining_to_move, remaining_to_ack, dma_tag_tmp, curr_len_move, curr_len_ack, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    configure_datapath(DATAPATH_DMA_REDUCTION, func, compression);

    remaining_to_move 	= len;
    remaining_to_ack 	= len;
    dma_tag_tmp 		= dma_tag;

    //default size of transaction is full size
    curr_len_move = dma_transaction_size;
    curr_len_ack  = dma_transaction_size;
    //1. issue at most max_dma_in_flight of dma_transaction_size
    for (i = 0; remaining_to_move > 0 && i < max_dma_in_flight ; i++){
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp  = start_dma(curr_len_move, op0_addr, rx_buff_addr, dst_addr, USE_OP0_DMA | USE_OP1_DMA | USE_RES_DMA, compression, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        op0_addr 			+= curr_len_move;
        if(compression)
            dst_addr 			+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
        else
            dst_addr 			+= curr_len_move;
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(remaining_to_move > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP0_DMA | USE_OP1_DMA | USE_RES_DMA, compression);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
        //enqueue other DMA movement
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, src_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp  = start_dma(curr_len_move, op0_addr, rx_buff_addr, dst_addr, USE_OP0_DMA | USE_OP1_DMA | USE_RES_DMA, compression, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        op0_addr 			+= curr_len_move;
        if(compression)
            dst_addr 			+= curr_len_move*arcfg.compressed_elem_bits/arcfg.compressed_elem_bits;
        else
            dst_addr 			+= curr_len_move;
    }
    //3. finish ack the remaining
    while(remaining_to_ack > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP0_DMA | USE_OP1_DMA | USE_RES_DMA, compression);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
    }

    return COLLECTIVE_OP_SUCCESS;
}

//1) receives from a rank 
//2) sums with a a buffer
//3) result is sent to another rank
int fused_recv_reduce_send(
        unsigned int src_rank,
        unsigned int dst_rank,
        unsigned int len,
        unsigned int func,
        uint64_t op0_addr,
        unsigned int mpi_tag,
        unsigned int compression)
    {
    
    uint64_t rx_buff_addr;
    unsigned int buf_idx;
    unsigned int remaining_to_move, remaining_to_ack, dma_tag_tmp, curr_len_move, curr_len_ack, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    configure_datapath( use_tcp ? DATAPATH_OFFCHIP_TCP_REDUCTION : DATAPATH_OFFCHIP_UDP_REDUCTION, func, compression);

    remaining_to_move 	= len;
    remaining_to_ack 	= len;
    dma_tag_tmp 		= dma_tag;

    //default size of transaction is full size
    curr_len_move = dma_transaction_size;
    curr_len_ack  = dma_transaction_size;
    //1. issue at most max_dma_in_flight of dma_transaction_size
    //start pack
    start_packetizer_message(dst_rank, len, mpi_tag);
    for (i = 0; remaining_to_move > 0 && i < max_dma_in_flight ; i++){
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, op0_addr, rx_buff_addr, 0, USE_OP0_DMA | USE_OP1_DMA, compression, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        if(compression)
            op0_addr += curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;   
        else
            op0_addr += curr_len_move;
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(remaining_to_move > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP0_DMA | USE_OP1_DMA, compression);
        //ack pack
        ack_packetizer(dst_rank, 1);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
        //enqueue other DMA movement
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //start pack
        start_packetizer_message(dst_rank, curr_len_move, mpi_tag);
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, op0_addr, rx_buff_addr, 0, USE_OP0_DMA | USE_OP1_DMA, compression, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
        if(compression)
            op0_addr += curr_len_move*arcfg.uncompressed_elem_bits/arcfg.compressed_elem_bits;   
        else
            op0_addr += curr_len_move;
    }
    //3. finish ack the remaining
    while(remaining_to_ack > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP0_DMA | USE_OP1_DMA, compression);
        //ack pack
        ack_packetizer(dst_rank, 1);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
    }

    return COLLECTIVE_OP_SUCCESS;
}

//receives from a rank and forwards to another rank 
int relay(
        unsigned int src_rank,
        unsigned int dst_rank,
        unsigned int len, 
        unsigned int mpi_tag){
    uint64_t rx_buff_addr;
    unsigned int buf_idx;
    unsigned int remaining_to_move, remaining_to_ack, dma_tag_tmp, curr_len_move, curr_len_ack, i;
    rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

    circular_buffer spare_buffer_queue;
    cb_init(&spare_buffer_queue, max_dma_in_flight);

    //configure datapath
    configure_datapath( use_tcp ? DATAPATH_OFFCHIP_TX_TCP: DATAPATH_OFFCHIP_TX_UDP, 0, NO_COMPRESSION);

    remaining_to_move 	= len;
    remaining_to_ack 	= len;
    dma_tag_tmp 		= dma_tag;

    //default size of transaction is full size
    curr_len_move = dma_transaction_size;
    curr_len_ack  = dma_transaction_size;
    //1. issue at most max_dma_in_flight of dma_transaction_size
    //start pack
    start_packetizer_message(dst_rank, len, mpi_tag);
    for (i = 0; remaining_to_move > 0 && i < max_dma_in_flight ; i++){
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, rx_buff_addr, 0, 0, USE_OP0_DMA, USE_NONE, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
    }
    //2.ack 1 and issue another dma transfer up until there's no more dma move to issue
    while(remaining_to_move > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP1_DMA, USE_NONE);
        //ack pack
        ack_packetizer(dst_rank, 1);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
        //enqueue other DMA movement
        if (remaining_to_move < dma_transaction_size){
            curr_len_move 	= remaining_to_move ;
        }
        //start pack
        start_packetizer_message(dst_rank, curr_len_move, mpi_tag);
        //wait for segment to come
        buf_idx = wait_on_rx(src_rank, curr_len_move, mpi_tag);
        if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
        rx_buff_addr = ((uint64_t) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
        //start DMAs
        dma_tag_tmp 		 = start_dma(curr_len_move, rx_buff_addr, 0, 0, USE_OP0_DMA, USE_NONE, dma_tag_tmp);
        //save spare buffer id
        cb_push(&spare_buffer_queue, buf_idx);
        remaining_to_move 	-= curr_len_move;
    }
    //3. finish ack the remaining
    while(remaining_to_ack > 0){
        if (remaining_to_ack < dma_transaction_size)
            curr_len_ack = remaining_to_ack;
        //wait for DMAs to finish
        ack_dma(curr_len_ack, USE_OP1_DMA, USE_NONE);
        //ack pack
        ack_packetizer(dst_rank, 1);
        //set spare buffer as free
        buf_idx = cb_pop(&spare_buffer_queue);
        microblaze_disable_interrupts();
        rx_buf_list[buf_idx].status = STATUS_IDLE;
        microblaze_enable_interrupts();
        remaining_to_ack  -= curr_len_ack;
    }

    return COLLECTIVE_OP_SUCCESS;
}

//COLLECTIVES

//root read multiple times the same segment and send it to each rank before moving to the next part of the buffer to be transmitted. 
int broadcast(  unsigned int len,
                unsigned int src_rank,
                uint64_t buf_addr,
                unsigned int compression){
    unsigned int curr_len, i;
    //determine if we're sending or receiving
    if(src_rank == world.local_rank){
        
        //configure datapath
        configure_datapath( use_tcp ? DATAPATH_OFFCHIP_TX_TCP: DATAPATH_OFFCHIP_TX_UDP, 0, compression);

        //send to all members of the communicator 1 segment of the buffer at the time
        curr_len = dma_transaction_size;
        while (len > 0)
        {
            if(len < dma_transaction_size)
                curr_len = len;
            for(i=0; i<world.size; i++){
                if(i == world.local_rank) continue;
                //start packetizer messages
                start_packetizer_message(i, curr_len, TAG_ANY);
                //issue a DMA command for each rank, with casting since this is pulling from source
                move_data(curr_len, buf_addr, 0, 0, USE_OP0_DMA, compression, 0, 0);
            }
            //ack packetizer 
            for(i=0; i<world.size; i++){
                if(i== world.local_rank) continue;
                
                ack_packetizer(i, 1);
            } 
            //move on in the buffer
            buf_addr += curr_len;
            len 	 -= curr_len;
        }
        return COLLECTIVE_OP_SUCCESS;
    }else{
        return recv(src_rank, len, buf_addr, TAG_ANY, compression);
    }
    
}

//scatter segment at a time. root sends each rank a segment in a round robin fashion 
int scatter(unsigned int len,
            unsigned int src_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int compression){

    unsigned int curr_len, i, offset_next_rank_buffer;
    uint64_t tmp_src_buf_base_addr, tmp_src_buf_offset_addr;
    offset_next_rank_buffer = len;

    //determine if we're sending or receiving
    if(src_rank == world.local_rank){
        
        //configure datapath
        configure_datapath( use_tcp ? DATAPATH_OFFCHIP_TX_TCP: DATAPATH_OFFCHIP_TX_UDP, 0, compression);

        tmp_src_buf_base_addr = src_buf_addr;

        //send to all members of the communicator 1 segment of the buffer at the time
        //w.r.t. broadcast the part that has to be sent is to i-esim_rank is [i*len+:len]
        curr_len = dma_transaction_size;
        while (len > 0)
        {
            if(len < dma_transaction_size)
                curr_len = len;
                
            tmp_src_buf_offset_addr = tmp_src_buf_base_addr;
            for(i=0; i<world.size; i++){
                if(i != world.local_rank){
                    //start packetizer messages and issue a single dma command for eachrank
                    start_packetizer_message(i, curr_len, TAG_ANY);
                    move_data(curr_len, tmp_src_buf_offset_addr, 0, 0, USE_OP0_DMA, compression, 0, 0);
                }
                tmp_src_buf_offset_addr += offset_next_rank_buffer;
            }
            //ack packetizer 
            for(i=0; i<world.size; i++){
                if(i== world.local_rank) continue;
                
                ack_packetizer(i, 1);
            } 
            //move on in the buffer
            tmp_src_buf_base_addr += curr_len;
            len 	 	 		  -= curr_len;
        }

        //for root copy
        src_buf_addr += world.local_rank*offset_next_rank_buffer;
        copy(offset_next_rank_buffer, src_buf_addr, dst_buf_addr, compression);

        return COLLECTIVE_OP_SUCCESS;
    }else{
        return recv(src_rank, len, dst_buf_addr, TAG_ANY, compression);
    }

}

//naive gather: non root relay data to the root. root copy segments in dst buffer as they come.
int gather( unsigned int len,
            unsigned int root_rank,
            uint64_t src_buf_addr,
            uint64_t dst_buf_addr,
            unsigned int compression){
    uint64_t tmp_buf_addr;
    unsigned int i,curr_pos, next_in_ring, prev_in_ring, number_of_shift;
    int ret = COLLECTIVE_OP_SUCCESS;

    next_in_ring = (world.local_rank + 1			  ) % world.size	;
    prev_in_ring = (world.local_rank + world.size - 1 ) % world.size	;
    
    if(root_rank == world.local_rank){ //root ranks mainly receives

        //receive from all members of the communicator
        for(i=0, curr_pos = prev_in_ring; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
            //TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
            //currently this will go sequentially through the segments, which is slow and also requires more spare RX buffers
            tmp_buf_addr  = dst_buf_addr + len * curr_pos;
            if(curr_pos==world.local_rank)
            { // root copies
                ret	= copy(len, src_buf_addr, tmp_buf_addr, compression);
            }else{
                ret = recv(prev_in_ring, len, tmp_buf_addr, TAG_ANY,compression);
            }
        }
    }else{
        //non root ranks sends their data + relay others data to the next rank in sequence
        // as a daisy chain
        number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ; //distance to the root
        ret += send(next_in_ring, len, src_buf_addr, TAG_ANY, compression);
        for (int i = 0; i < number_of_shift; i++)
        {	
            //relay the others 
            relay(prev_in_ring, next_in_ring, len, TAG_ANY);
        }	
    }
    return ret;
}

//naive fused: 1) receive a segment 2) move in the dest buffer 3) relay to next rank 
int allgather(  unsigned int len,
                uint64_t src_buf_addr,
                uint64_t dst_buf_addr,
                unsigned int compression){
    uint64_t tmp_buf_addr;
    unsigned int i,curr_pos, next_in_ring, prev_in_ring;
    int ret = COLLECTIVE_OP_SUCCESS;

    next_in_ring = (world.local_rank+1			   ) % world.size	;
    prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
    //load buf_addr

    //send our data to next in ring
    ret += send(next_in_ring, len, src_buf_addr, TAG_ANY, compression);
    //receive from all members of the communicator
    for(i=0, curr_pos = world.local_rank; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
        tmp_buf_addr  = dst_buf_addr + len * curr_pos;

        if(curr_pos==world.local_rank){
            ret	= copy(len, src_buf_addr, tmp_buf_addr, compression);
        }else{
            ret = recv(prev_in_ring, len, tmp_buf_addr, TAG_ANY, compression);
            //TODO: use the same stream to move data both to dst_buf_addr and tx_subsystem 
            //todo: use DMA1 RX to forward to next and DMA0/2 RX and DMA1 TX to copy ~ at the same time! 
            if(i+1 < world.size){ //if not the last data needed relay to the next in the sequence
                ret = send(next_in_ring, len, tmp_buf_addr, TAG_ANY, compression);
            }
        }		
    }

    return ret;
}

//every rank receives a buffer it reduces its own buffer and forwards to next rank in the ring
int reduce( unsigned int len,
            unsigned int func,
            unsigned int root_rank,
            uint64_t src_addr,
            uint64_t dst_addr,
            unsigned int compression){
    int ret;

    unsigned int next_in_ring = (world.local_rank+1			   ) % world.size	;
    unsigned int prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
    //determine if we're sending or receiving
    if( prev_in_ring == root_rank){ 
        //non root ranks immediately after the root sends
        ret = send(next_in_ring, len, src_addr, TAG_ANY, compression);
    }else if (world.local_rank != root_rank){
        //non root ranks sends their data + data received from previous rank to the next rank in sequence as a daisy chain
        ret = fused_recv_reduce_send(prev_in_ring, next_in_ring, len, func, src_addr,  TAG_ANY, compression);
    }else{	
        //root only receive from previous node in the ring, add its local buffer and save in destination buffer 
        ret = fused_recv_reduce(prev_in_ring, len, func, src_addr, dst_addr, TAG_ANY, compression);
    }
    return ret;
}

//scatter_reduce: (a,b,c), (1,2,3), (X,Y,Z) -> (a+1+X,,) (,b+2+Y,) (,,c+3+Z)
//len == size of chunks
int scatter_reduce( unsigned int len,
                    unsigned int func,
                    uint64_t src_addr,
                    uint64_t dst_addr,
                    unsigned int compression){
    unsigned int ret,i;
    uint64_t curr_recv_addr, curr_send_addr;

    unsigned int next_in_ring 	 = (world.local_rank + 1			) % world.size	;
    unsigned int prev_in_ring 	 = (world.local_rank + world.size-1 ) % world.size	;
    unsigned int curr_send_chunk = prev_in_ring;
    unsigned int curr_recv_chunk = (prev_in_ring + world.size-1 ) % world.size	;

    //1. each rank send its own chunk to the next rank in the ring
    curr_send_addr = src_addr + len * curr_send_chunk;
    ret = send(next_in_ring, len, curr_send_addr, TAG_ANY, compression);
    if(ret != COLLECTIVE_OP_SUCCESS) return ret;
    //2. for n-1 times sum the chunk that is coming from previous rank with your chunk at the same location. then forward the result to
    // the next rank in the ring
    for (i = 0; i < world.size -2; ++i, curr_recv_chunk=(curr_recv_chunk + world.size - 1) % world.size)
    {
        curr_recv_addr = src_addr + len * curr_recv_chunk;
        //2. receive part of data from previous in rank and accumulate to the part you have at the same address
        //ad forwar dto the next rank in the ring
        ret = fused_recv_reduce_send(prev_in_ring, next_in_ring, len, func, curr_recv_addr, TAG_ANY, compression); 
        if(ret != COLLECTIVE_OP_SUCCESS) return ret;
    }
    //at last iteration (n-1) you can sum and save the result at the same location (which is the right place to be)
    if (i > 0 ){
        curr_recv_addr  = src_addr + len * curr_recv_chunk; 
        curr_send_addr  = dst_addr + len * curr_recv_chunk;
        ret = fused_recv_reduce(prev_in_ring, len, func, curr_recv_addr, curr_send_addr, TAG_ANY, compression); 
        if(ret != COLLECTIVE_OP_SUCCESS) return ret;
    }
    return COLLECTIVE_OP_SUCCESS;
}

//2 stage allreduce: distribute sums across the ranks. smaller bandwidth between ranks and use multiple arith at the same time. scatter_reduce+all_gather
int allreduce(  unsigned int len,
                unsigned int func,
                uint64_t src_addr,
                uint64_t dst_addr,
                unsigned int compression){
    unsigned int ret,i;
    uint64_t curr_recv_addr, curr_send_addr;

    int curr_send_chunk	, curr_recv_chunk			  = world.local_rank;			 
    //divide into chunks 
    unsigned int len_div_size 	 	= len/world.size; 
    unsigned int min_operand_width  = arcfg.compressed_elem_bits/8;//TODO: what if the arith op is not a multiple of 1 byte?
    unsigned int tail				= len_div_size % min_operand_width;
    len_div_size += - tail + (tail == 0 ? 0 : min_operand_width); //TODO: now when processing last chunk it will assume that buffers are same length and part of non buffer will be processed.
    unsigned int next_in_ring 	 = (world.local_rank+1			  )%world.size	;
    unsigned int prev_in_ring 	 = (world.local_rank+world.size-1 )%world.size	;
    //scatter reduce
    scatter_reduce(len_div_size, func, src_addr, dst_addr, compression);
    //allgather in place on dst_buffer
    for (i = 0; i < world.size -1; ++i)
    {
        curr_send_chunk =  curr_recv_chunk;
        curr_recv_chunk = (curr_recv_chunk + world.size - 1) % world.size;
        curr_send_addr  = dst_addr + len_div_size * curr_send_chunk;
        curr_recv_addr  = dst_addr + len_div_size * curr_recv_chunk;
        //TODO: 5.6. can be done together since we have 3 dmas (1rx read from spare, 1tx write in the final dest, 2rx can read and transmit to next in the ring)
        //5. send the local reduced results to next rank
        ret = send(next_in_ring, len_div_size, curr_send_addr, TAG_ANY, compression);
        if(ret != COLLECTIVE_OP_SUCCESS) return ret;
        //6. receive the reduced results from previous rank and put into correct dst buffer position
        ret = recv(prev_in_ring, len_div_size, curr_recv_addr, TAG_ANY, compression);
        if(ret != COLLECTIVE_OP_SUCCESS) return ret;
    }

    return COLLECTIVE_OP_SUCCESS;
}

int ext_kernel_push(unsigned int len,
                    unsigned int krnl_id,
                    uint64_t src_addr,
                    unsigned int compression
){
    //configure switch
    configure_datapath(DATAPATH_DMA_LOOPBACK, 0, compression);
    return move_data(len, src_addr, 0, 0, USE_OP0_DMA, compression, 0, 0);
}

int ext_kernel_pull(unsigned int len,
                    unsigned int krnl_id,
                    uint64_t dst_addr,
                    unsigned int compression
){
    //configure switch
    configure_datapath(DATAPATH_DMA_LOOPBACK, 0, compression);
    return move_data(len, 0, 0, dst_addr, USE_RES_DMA, compression, 0, 0);
}

//startup and main

void check_hwid(void){
    // read HWID from hardware and copy it to host-accessible memory
    // TODO: check the HWID against expected
    unsigned int hwid = Xil_In32(GPIO2_DATA_REG);
    Xil_Out32(HWID_OFFSET, hwid);
}

//initialize the system
void init(void) {
    int myoffset;
    // Register irq handler
    microblaze_register_handler((XInterruptHandler)stream_isr,(void*)0);
    microblaze_enable_interrupts();

    // initialize exchange memory to zero.
    for ( myoffset = EXCHMEM_BASEADDR; myoffset < END_OF_EXCHMEM; myoffset +=4) {
        Xil_Out32(myoffset, 0);
    }
    // Check hardware ID
    check_hwid();
    //initialize dma tag
    dma_tag = 0;
    for(int i=0; i < MAX_DMA_TAGS; i++){
        dma_tag_lookup[i]=-1;
    }
    //deactivate reset of all peripherals
    SET(GPIO_DATA_REG, GPIO_SWRST_MASK);
    //enable access from host to exchange memory by removing reset of interface
    SET(GPIO_DATA_REG, GPIO_READY_MASK);

}

//reset the control module
//since this cancels any dma movement in flight and 
//clears the queues it's necessary to reset the dma tag and dma_tag_lookup
void encore_soft_reset(void){
    int myoffset;
    //disable interrupts (since this procedure will toggle the irq pin)
    microblaze_disable_interrupts();
    setup_irq(0);
    //1. activate reset pin  
    CLR(GPIO_DATA_REG, GPIO_SWRST_MASK);
    //2. clean up (exchange memory and static variables)
    for ( myoffset = EXCHMEM_BASEADDR; myoffset < END_OF_EXCHMEM; myoffset +=4) {
        Xil_Out32(myoffset, 0);
    }
    num_rx_enqueued = 0;
    dma_tag = 0;
    //next_rx_tag = 0;
    //clear dma lookup table
    for(int i=0; i < MAX_DMA_TAGS; i++){
        dma_tag_lookup[i]=-1;
    }
    //3. recheck hardware ID
    check_hwid();
    //4. then deactivate reset of all other blocks
    SET(GPIO_DATA_REG, GPIO_SWRST_MASK);
}

//poll for a call from the host
static inline void wait_for_call(void) {
    // Poll the host cmd queue
    unsigned int invalid;
    do {
        invalid = 0;
        invalid += tngetd(CMD_HOST);
    } while (invalid);
}

//signal finish to the host and write ret value in exchange mem
static inline void finalize_call(unsigned int retval) {
    Xil_Out32(RETVAL_OFFSET, retval);
    // Done: Set done and idle
    putd(STS_HOST, retval);
}

int main() {
    unsigned int retval;
    unsigned int scenario, len, comm, root_src_dst, function, msg_tag;
    unsigned int datapath_cfg, compression_flags, stream_flags;
    unsigned int op0_addrl, op0_addrh, op1_addrl, op1_addrh, res_addrl, res_addrh;
    uint64_t op0_addr, op1_addr, res_addr;

    init();
    //register exception handler though setjmp. it will save stack status to unroll stack when the jmp is performed 
    //	ref: https://www.gnu.org/software/libc/manual/html_node/Longjmp-in-Handler.html
    // when first executed, setjmp returns 0. call to longjmp pass a value != 0 to indicate a jmp is occurred
    retval = setjmp(excp_handler);
    if(retval)
    {
        finalize_call(retval);
    }

    while (1) {
        wait_for_call();
        
        //read parameters from host command queue
        scenario            = getd(CMD_HOST);
        len                 = getd(CMD_HOST);
        comm                = getd(CMD_HOST);
        root_src_dst        = getd(CMD_HOST);
        function            = getd(CMD_HOST);
        msg_tag             = getd(CMD_HOST);
        datapath_cfg        = getd(CMD_HOST);
        compression_flags   = getd(CMD_HOST);
        stream_flags        = getd(CMD_HOST);
        op0_addrl           = getd(CMD_HOST);
        op0_addrh           = getd(CMD_HOST);
        op1_addrl           = getd(CMD_HOST);
        op1_addrh           = getd(CMD_HOST);
        res_addrl           = getd(CMD_HOST);
        res_addrh           = getd(CMD_HOST);

        op0_addr = ((uint64_t) op0_addrh << 32) | op0_addrl;
        op1_addr = ((uint64_t) op1_addrh << 32) | op1_addrl;
        res_addr = ((uint64_t) res_addrh << 32) | res_addrl;

        //initialize arithmetic/compression config and communicator
        //NOTE: these are global because they're used in a lot of places but don't change during a call
        //TODO: determine if they can remain global in hierarchical collectives
        arcfg = *((datapath_arith_config *)datapath_cfg);
        world = find_comm(comm);
        
        switch (scenario)
        {
            case XCCL_CONFIG:
                retval = 0;
                switch (function)
                {
                    case HOUSEKEEP_IRQEN:
                        setup_irq((use_tcp ? (IRQCTRL_DMA2_CMD_QUEUE_EMPTY|IRQCTRL_DMA2_STS_QUEUE_NON_EMPTY) : (IRQCTRL_DMA0_CMD_QUEUE_EMPTY|IRQCTRL_DMA0_STS_QUEUE_NON_EMPTY)));
                        microblaze_enable_interrupts();
                        break;
                    case HOUSEKEEP_IRQDIS:
                        microblaze_disable_interrupts();
                        setup_irq(0);
                        clear_timer_interrupt();
                        break;
                    case HOUSEKEEP_SWRST:
                        encore_soft_reset();
                        break;
                    case HOUSEKEEP_PKTEN:
                        start_depacketizer(UDP_RXPKT_BASEADDR);
                        start_depacketizer(TCP_RXPKT_BASEADDR);
                        start_packetizer(UDP_TXPKT_BASEADDR, MAX_PACKETSIZE);
                        start_packetizer(TCP_TXPKT_BASEADDR, MAX_PACKETSIZE);
                        break;
                    case HOUSEKEEP_TIMEOUT:
                        timeout = len;
                        break;
                    case INIT_CONNECTION:
                        retval  = init_connection();
                        break;
                    case OPEN_PORT:
                        retval  = openPort();
                        break;
                    case OPEN_CON:
                        retval  = openCon();
                        break;
                    case USE_TCP_STACK:
                        use_tcp = 1;
                        break;
                    case USE_UDP_STACK:
                        use_tcp = 0;
                    case START_PROFILING:
                        retval = COLLECTIVE_OP_SUCCESS;
                        break;
                    case END_PROFILING:
                        retval = COLLECTIVE_OP_SUCCESS;
                        break;
                    case SET_DMA_TRANSACTION_SIZE:
                        retval = DMA_NOT_EXPECTED_BTT_ERROR;
                        if(len < DMA_MAX_BTT){
                            dma_transaction_size = len;
                            retval = COLLECTIVE_OP_SUCCESS;
                        }
                        break;
                    case SET_MAX_DMA_TRANSACTIONS:
                        retval = DMA_NOT_OKAY_ERROR;
                        if(len < DMA_MAX_TRANSACTIONS){
                            max_dma_in_flight = len;
                            retval = COLLECTIVE_OP_SUCCESS;
                        }
                        break;
                    default:
                        break;
                }
                
                break;
            case XCCL_REDUCE:
                retval = reduce_loopback(len, function, op0_addr, op1_addr, res_addr, compression_flags);
                break;
            case XCCL_COPY:
                retval = copy(len, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_SEND:
                retval = send(root_src_dst, len, op0_addr, msg_tag, compression_flags);
                break;
            case XCCL_RECV:
                retval = recv(root_src_dst, len, res_addr, msg_tag, compression_flags);
                break;
            case XCCL_COLL_BCAST:
                retval = broadcast(len, root_src_dst, op0_addr, compression_flags);
                break;
            case XCCL_COLL_SCATTER:    
                retval = scatter(len, root_src_dst, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_COLL_GATHER:
                retval = gather(len, root_src_dst, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_COLL_REDUCE:
                retval = reduce(len, function, root_src_dst, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_COLL_ALLGATHER:
                retval = allgather(len, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_COLL_ALLREDUCE:
                retval = allreduce(len, function, op0_addr, res_addr, compression_flags);
                break;
            case XCCL_COLL_REDUCE_SCATTER:
                retval = scatter_reduce(len, function, op0_addr, res_addr, compression_flags);
                break;
            default:
                retval = COLLECTIVE_OP_SUCCESS;
                break;
        }

        finalize_call(retval);
    }
    return 0;
}