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
#define OPT_SEG
#define OPT_SEG_INTERLEAVE_BCAST 
#define OPT_RING_ALL_REDUCE
#define MAX_SEG_SIZE 1048576
#define MAX_DMA_TRANSACTION_BYTE 4194304

static volatile int 		 use_tcp = 1;
static jmp_buf 				 excp_handler;
static volatile int 		 dma_tag;
static volatile unsigned int next_rx_tag 	 = 0;
static volatile unsigned int num_rx_enqueued = 0;
static volatile unsigned int timeout 	 	 = 1 << 28;
static volatile 		 int dma_tag_lookup [MAX_DMA_TAGS]; //index of the spare buffer that has been issued with that dma tag. -1 otherwise

unsigned int enqueue_rx_buffers(void);
int  dequeue_rx_buffers(void);

void check_hwid(void){
	// read HWID from hardware and copy it to host-accessible memory
	// TODO: check the HWID against expected
	unsigned int hwid = Xil_In32(GPIO2_DATA_REG);
	Xil_Out32(HWID_OFFSET, hwid);
}

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

//creates a timeout that will interrupt the MB code execution 
static inline void start_timeout(unsigned int time){
	//1. set timer load register x(TLRx) with time so that TIMING_INTERVAL = (TLRx + 2) * AXI_CLOCK_PERIOD
	//2. load timer register
	//3. set counter enable/down/non-autoreload/interrupt enable/clear load /generate mode
	Xil_Out32(TIMER_BASEADDR + TIMER0_LOAD_REGISTER_OFFSET, time);
	Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_LOAD_TIMER_MASK);
	Xil_Out32(TIMER_BASEADDR + TIMER0_CONTROL_AND_STATUS_REGISTER_OFFSET, CONTROL_AND_STATUS_REGISTER_INTERRUPT_MASK | CONTROL_AND_STATUS_REGISTER_ENABLE_MASK | CONTROL_AND_STATUS_REGISTER_INTERRUPT_ENABLE_MASK | CONTROL_AND_STATUS_REGISTER_UP_DOWN_MASK);
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

void start_arith(unsigned int byte_count, unsigned int function) {
	//get number of 512b/64B transfers corresponding to byte_count
	unsigned int ntransfers = (byte_count+63)/64;
	Xil_Out32(ARITH_BASEADDR+0x10, ntransfers);
	Xil_Out32(ARITH_BASEADDR+0x18, function);
	SET(ARITH_BASEADDR, CONTROL_START_MASK);
}

void start_packetizer(long long base_addr,unsigned int max_pktsize) {
	//get number of 512b/64B transfers corresponding to max_pktsize
	unsigned int max_pkt_transfers = (max_pktsize+63)/64;
	Xil_Out32(	base_addr +0x10	, max_pkt_transfers);
	SET(	 	base_addr 		, CONTROL_REPEAT_MASK);
	SET(		base_addr 		, CONTROL_START_MASK);
}

void start_depacketizer(long long base_addr) {
	SET(		base_addr, CONTROL_REPEAT_MASK);
	SET(		base_addr, CONTROL_START_MASK);
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
	if (( n_enqueued == 0 && (irq & (use_tcp ? IRQCTRL_DMA2_CMD_QUEUE_EMPTY : IRQCTRL_DMA0_CMD_QUEUE_EMPTY) )) || (irq & IRQCTRL_TIMER_ENABLE) )
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

//initialize the system

void init(void) {
	int myoffset;
	// Register irq handler
	microblaze_register_handler((XInterruptHandler)stream_isr,(void*)0);
	microblaze_enable_interrupts(); //TODO: check if we can remove

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
	//disable interrupts (since this will toggle the irq pin)
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
void wait_for_call(void) {
	// Poll the host cmd queue
	unsigned int invalid;
	do {
		invalid = 0;
		invalid += tngetd(CMD_HOST);
	} while (invalid);
}

//signal finish to the host
void finalize_call(unsigned int retval) {
	// Xil_Out32(EXCHMEM_BASEADDR+RETVAL_OFFSET, retval);
	Xil_Out32(RETVAL_OFFSET, retval);
    // Done: Set done and idle
	putd(STS_HOST, retval);
}

//start a DMA operations on a specific channel
void dma_cmd(unsigned int channel, unsigned int btt, unsigned int addrl, unsigned int addrh, unsigned int tag) {
	putd(channel, 0xC0800000 | btt); // 31=DRR 30=EOF 29-24=DSA 23=Type 22-0=BTT
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

//configure a switch path from a source to a destination
//PG085 page 26
//config registers start at 0x40
void set_switch_datapath(unsigned int base, unsigned int src, unsigned int dst) {
	Xil_Out32(base+0x40+4*dst, src);
}

//disable a switch master
//PG085 page 26
//config registers start at 0x40
void disable_switch_datapath(unsigned int base, unsigned int dst) {
	Xil_Out32(base+0x40+4*dst, 0x80000000);
}

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

//TODO: update description
//configure the switch for various scenarios:
//1 == DATAPATH_DMA_LOOPBACK	 : DMA1 RX -> DMA1 TX (DMA Loopback)
//2 == DATAPATH_DMA_REDUCTION	 : DMA0 RX -> Arith Op0 (DMA Reduction)
//   DMA1 RX -> Arith Op1 
//   Arith Res -> DMA1 TX 
//3 == DATAPATH_OFFCHIP_TX		 : DMA1 RX -> Eth TX (Direct Off-chip TX)
//4 == DATAPATH_OFFCHIP_REDUCTION: DMA0 RX -> Arith Op0 (Reduced Off-chip TX)
//   DMA1 RX -> Arith Op1 
//   Arith Res -> Eth TX
void cfg_switch(unsigned int scenario, unsigned int arith) {
	switch (scenario)
	{
		case DATAPATH_DMA_LOOPBACK:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP0);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP1);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);			
			break;
		case DATAPATH_DMA_REDUCTION:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA0_RX, MAIN_SWITCH_M_ARITH_OP0);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_ARITH_OP1);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_ARITH_RES, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);	
			break;
		case DATAPATH_OFFCHIP_TX_UDP:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP0);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP1);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);	
			break;
		case DATAPATH_OFFCHIP_TX_TCP:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP0);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP1);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);	
			break;
		case DATAPATH_OFFCHIP_UDP_REDUCTION:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA0_RX, MAIN_SWITCH_M_ARITH_OP0);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_ARITH_OP1);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_ARITH_RES, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);	
			break;
		case DATAPATH_OFFCHIP_TCP_REDUCTION:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA0_RX, MAIN_SWITCH_M_ARITH_OP0);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA1_RX, MAIN_SWITCH_M_ARITH_OP1);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_ARITH_RES, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_UDP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_EXT_KRNL);	
			break;
		case DATAPATH_DMA_EXT_LOOPBACK:
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_DMA0_RX, MAIN_SWITCH_M_EXT_KRNL);
			set_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_S_EXT_KRNL, MAIN_SWITCH_M_DMA1_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP0);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_ARITH_OP1);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_TCP_TX);
			disable_switch_datapath(MAIN_SWITCH_BASEADDR, MAIN_SWITCH_M_UDP_TX);
			break;//TODO: add DATAPATH_OFFCHIP_RX
		default:
			return;
	}
	apply_switch_config(MAIN_SWITCH_BASEADDR);
	switch (arith)
	{
		case ARITH_EXTERNAL:
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_OP0, ARITH_SWITCH_M_EXT_OP0);
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_OP1, ARITH_SWITCH_M_EXT_OP1);
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_EXT_RES, ARITH_SWITCH_M_RES);
			disable_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_M_INT_OP0);
			disable_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_M_INT_OP1);
			break;
		case ARITH_INTERNAL:
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_OP0, ARITH_SWITCH_M_INT_OP0);
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_OP1, ARITH_SWITCH_M_INT_OP1);
			set_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_S_INT_RES, ARITH_SWITCH_M_RES);
			disable_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_M_EXT_OP0);
			disable_switch_datapath(ARITH_SWITCH_BASEADDR, ARITH_SWITCH_M_EXT_OP1);
			break;
		default:

			break;
	}
	apply_switch_config(ARITH_SWITCH_BASEADDR);
}

//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
int dma_loopback(	unsigned int len,
					unsigned int src_addrl,
					unsigned int src_addrh,
					unsigned int dst_addrl,
					unsigned int dst_addrh) {
	int dma_tag_tmp;
	unsigned int invalid, status, count = 0;
	//configure switch
	if(cfg_switch(DATAPATH_DMA_LOOPBACK, ARITH_INTERNAL )) return CONFIG_SWITCH_ERROR;

	int num_transaction = (len + MAX_DMA_TRANSACTION_BYTE - 1) / MAX_DMA_TRANSACTION_BYTE;

	uint64_t curr_src_addr = (uint64_t)src_addrl | ( (uint64_t)src_addrh << 32);
	unsigned int curr_src_addrl, curr_src_addrh;

	uint64_t curr_dst_addr = (uint64_t)dst_addrl | ( (uint64_t)dst_addrh << 32);
	unsigned int curr_dst_addrl, curr_dst_addrh;

	unsigned int curr_len = MAX_DMA_TRANSACTION_BYTE;
	unsigned int sent_len = 0;

	//issue dma commands
	dma_tag_tmp = dma_tag;
	for (int i = 0; i < num_transaction; ++i)
	{
		if (sent_len + MAX_DMA_TRANSACTION_BYTE > len)
			curr_len = len - sent_len;
		else
			curr_len = MAX_DMA_TRANSACTION_BYTE;
		curr_src_addrl = curr_src_addr & 0xFFFFFFFF;
		curr_src_addrh = (curr_src_addr >> 32) & 0xFFFFFFFF;
		curr_dst_addrl = curr_dst_addr & 0xFFFFFFFF;
		curr_dst_addrh = (curr_dst_addr >> 32) & 0xFFFFFFFF;

		//start both channels of the DMA1
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf);
		dma_cmd(CMD_DMA1_RX, curr_len, curr_src_addrl, curr_src_addrh, dma_tag_tmp);
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf);
		dma_cmd(CMD_DMA1_TX, curr_len, curr_dst_addrl, curr_dst_addrh, dma_tag_tmp);

		curr_src_addr = curr_src_addr + curr_len;
		curr_dst_addr = curr_dst_addr + curr_len;
		sent_len = sent_len + curr_len;
	}

	//Collect dma response
	for (int i = 0; i < num_transaction; ++i)
	{
		//wait for both channels to finish
		unsigned int invalid, status;
		unsigned int count = 0;
		do {
			if(timeout != 0 && count >= timeout )
				return -1;
			count ++;
			invalid = 0;
			invalid += tngetd(STS_DMA1_RX);
			invalid += tngetd(STS_DMA1_TX);
		} while (invalid);
		status = getd(STS_DMA1_RX);
		dma_tag = (dma_tag + 1) & 0xf);
		check_DMA_status(status, len, dma_tag, 0);
		status = getd(STS_DMA1_TX);
		dma_tag = (dma_tag + 1) & 0xf);
		check_DMA_status(status, len, dma_tag, 1);
		
	}	

	return COLLECTIVE_OP_SUCCESS;
}

//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
static inline int copy(	unsigned int len,
						unsigned int src_addrl,
						unsigned int src_addrh,
						unsigned int dst_addrl,
						unsigned int dst_addrh) {
	return  dma_loopback(	  len, src_addrl, src_addrh, dst_addrl, dst_addrh);
}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx writes back to dst buffer
int reduce_loopback(	unsigned int len,
						unsigned int func,
						unsigned int op1_addrl,
						unsigned int op1_addrh,
						unsigned int op2_addrl,
						unsigned int op2_addrh,
						unsigned int dst_addrl,
						unsigned int dst_addrh) {
	int dma_tag1,dma_tag2, dma_tag3; 
	//configure switch
	cfg_switch(DATAPATH_DMA_REDUCTION, ARITH_INTERNAL);
	//start arith hls ip
	start_arith(len, func);
	//start both channels of the DMA1
	dma_tag1 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA0_RX, len, op1_addrl, op1_addrh, dma_tag);
	dma_tag2 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA1_RX, len, op2_addrl, op2_addrh, dma_tag);
	dma_tag3 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA1_TX, len, dst_addrl, dst_addrh, dma_tag);
	//wait for all 3 channels to finish
	unsigned int invalid, status;
	unsigned int count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		invalid += tngetd(STS_DMA0_RX);
		invalid += tngetd(STS_DMA1_RX);
		invalid += tngetd(STS_DMA1_TX);
	} while (invalid);
	status = getd(STS_DMA0_RX);
	check_DMA_status(status, len, dma_tag1, 0);
	status = getd(STS_DMA1_RX);
	check_DMA_status(status, len, dma_tag2, 0);
	status = getd(STS_DMA1_TX);
	check_DMA_status(status, len, dma_tag3, 1);
	//TODO: check status of arith ip
	return COLLECTIVE_OP_SUCCESS;
}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx overwrites op2 buffer
static inline int accumulate (	
						unsigned int len,
						unsigned int func,
						unsigned int op1_addrl,
						unsigned int op1_addrh,
						unsigned int op2_addrl,
						unsigned int op2_addrh,
						unsigned int dst_addrl,
						unsigned int dst_addrh) {
	return reduce_loopback(	 len, func, op1_addrl, op1_addrh, op2_addrl, op2_addrh, dst_addrl, dst_addrh);
}

//retrieves the communicator
communicator find_comm(unsigned int adr){
	communicator ret;
	//TODO: it's not enough to set ret address?
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



int transmit_offchip_cmd(
						communicator *world,
						unsigned int dst_rank,
						unsigned int len,
						unsigned int src_addrl,
						unsigned int src_addrh,
						unsigned int dst_tag)
{
	unsigned int src_rank,sequence_number, dst, net_tx;
	//prepare message header arguments
	src_rank		= world->local_rank;
	sequence_number = world->ranks[dst_rank].outbound_seq;
	if (use_tcp){
	net_tx 			= CMD_TCP_TX; 
	dst 			= world->ranks[dst_rank].session; 
	cfg_switch( DATAPATH_OFFCHIP_TX_TCP , ARITH_INTERNAL);
	}else{
	net_tx		 	= CMD_UDP_TX; 
	cfg_switch( DATAPATH_OFFCHIP_TX_UDP , ARITH_INTERNAL);
	dst		 		= world->ranks[dst_rank].port; 
	}

	putd(net_tx, dst);
	putd(net_tx, len);
	putd(net_tx, dst_tag);
	putd(net_tx, src_rank);
	cputd(net_tx,sequence_number);
	//start RX channel of DMA1
	dma_tag = (dma_tag + 1) & 0xf;
	dma_cmd(CMD_DMA1_RX, len, src_addrl, src_addrh, dma_tag);

	//finally update seq number
	world->ranks[dst_rank].outbound_seq ++;

	return 0;
}

int transmit_offchip_sts()
{
	//wait for both channels to finish
	unsigned int invalid, status;
	do {
		invalid = 0;
		invalid += tngetd(STS_DMA1_RX);
	} while (invalid);
	status = getd(STS_DMA1_RX);// S2MM: 15=OKAY 14=SLVERR 13=DECERR 12=INTERR 11-8=TAG MM2S: 7=OKAY 6=SLVERR 5=DECERR 4=INTERR 3-0=TAG
	//TODO: check status - okay, tags
	return 0;
}

#ifdef OPT_SEG

//transmits a buffer to a rank of the world communicator
int transmit_offchip_world(
	communicator *world,
	unsigned int dst_rank,
	unsigned int len,
	unsigned int src_addrl,
	unsigned int src_addrh,
	unsigned int dst_tag
){

	int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
	uint64_t curr_addr = (uint64_t)src_addrl | ( (uint64_t)src_addrh << 32);
	unsigned int curr_addrl, curr_addrh;
	unsigned int curr_len = MAX_SEG_SIZE;
	unsigned int sent_len = 0;

	//configure switch
	if (use_tcp){
		cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_INTERNAL);
	}else{
		cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_INTERNAL);
	}

	//Send dma command
	//TODO: avoid cmd queue overflow
	for (int i = 0; i < num_segment; ++i)
	{
		if (sent_len + MAX_SEG_SIZE > len)
			curr_len = len - sent_len;
		else
			curr_len = MAX_SEG_SIZE;
		curr_addrl = curr_addr & 0xFFFFFFFF;
		curr_addrh = (curr_addr >> 32) & 0xFFFFFFFF;
		transmit_offchip_cmd(world, dst_rank, curr_len, curr_addrl, curr_addrh, dst_tag);
		curr_addr = curr_addr + curr_len;
		sent_len = sent_len + curr_len;
	}

	//Collect dma response
	for (int i = 0; i < num_segment; ++i)
	{
		transmit_offchip_sts();
	}	
	return 0;
}

#else
//transmits a buffer to a rank of the world communicator
int transmit_offchip_world(
	communicator *world,
	unsigned int dst_rank,
	unsigned int len,
	unsigned int src_addrl,
	unsigned int src_addrh,
	unsigned int dst_tag
){
	unsigned int src_rank,sequence_number, dst, net_tx;
	//prepare message header arguments
	src_rank		= world->local_rank;
	if (use_tcp){
	dst 			= world->ranks[dst_rank].session; 
	}else{
	dst		 		= world->ranks[dst_rank].port; 
	}
	sequence_number = world->ranks[dst_rank].outbound_seq + 1;
	//configure switch
	//send command to packetizer
	if (use_tcp){
	net_tx 			= CMD_TCP_TX; 
	cfg_switch( DATAPATH_OFFCHIP_TX_TCP , ARITH_INTERNAL);
	}else{
	net_tx		 	= CMD_UDP_TX; 
	cfg_switch( DATAPATH_OFFCHIP_TX_UDP , ARITH_INTERNAL);
	}
	putd(net_tx, dst); //start packetizer
	putd(net_tx, len);
	putd(net_tx, dst_tag);
	putd(net_tx, src_rank);
	cputd(net_tx,sequence_number);
	//start RX channel of DMA1
	dma_tag = (dma_tag + 1) & 0xf;
	dma_cmd(CMD_DMA1_RX, len, src_addrl, src_addrh, dma_tag);
	//wait for both channels to finish
	unsigned int invalid, status;
	unsigned int count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		invalid += tngetd(STS_DMA1_RX);
	} while (invalid);
	status = getd(STS_DMA1_RX);
	check_DMA_status(status, len, dma_tag, 0);
	//finally update seq number
	world->ranks[dst_rank].outbound_seq ++;
	return COLLECTIVE_OP_SUCCESS;
}
#endif

//transmits a buffer to a rank of the comm-esim communicator 
static inline int transmit_offchip(	
						unsigned int comm,
						unsigned int dst_rank,
						unsigned int len,
						unsigned int src_addrl,
						unsigned int src_addrh,
						unsigned int dst_tag
					) {
	//find communicator
	communicator world = find_comm(comm);
	return transmit_offchip_world(&world, dst_rank, len, src_addrl, src_addrh, dst_tag);
}

//performs an accumulate using DMA0 and DMA1 and then moves the result 
//through tx_subsystem to dst_rank
int reduce_offchip_world(
	communicator *world,
	unsigned int dst_rank,
	unsigned int len, 
	unsigned int func,
	unsigned int op0_addrl,
	unsigned int op0_addrh,
	unsigned int op1_addrl,
	unsigned int op1_addrh,
	unsigned int dst_tag
){
	unsigned int src_rank, dst, sequence_number, net_tx;
	unsigned int dma_tag1, dma_tag2;
	unsigned int invalid, status, count = 0;
	//prepare header arguments
	src_rank 		= world->local_rank;
	sequence_number = world->ranks[dst_rank].outbound_seq + 1;
	if (use_tcp){
	dst 			= world->ranks[dst_rank].session; 
	cfg_switch(DATAPATH_OFFCHIP_TCP_REDUCTION, ARITH_EXTERNAL);
	}else{
	dst		 		= world->ranks[dst_rank].port; 
	cfg_switch(DATAPATH_OFFCHIP_UDP_REDUCTION, ARITH_INTERNAL);
	}
	//configure switch
	//start arith hls ip
	start_arith(len, func);
	//send command to packetizer
	if (use_tcp){
	net_tx 			= CMD_TCP_TX; 
	}else{
	net_tx		 	= CMD_UDP_TX; 
	}
	putd(net_tx, dst);
	putd(net_tx, len);
	putd(net_tx, dst_tag);
	putd(net_tx, src_rank);
	cputd(net_tx,sequence_number);
	//start both channels of the DMA1
	dma_tag1 = dma_tag = (dma_tag + 1) & 0xf;
	dma_cmd(CMD_DMA0_RX, len, op0_addrl, op0_addrh, dma_tag);
	dma_tag2 = dma_tag = (dma_tag + 1) & 0xf;
	dma_cmd(CMD_DMA1_RX, len, op1_addrl, op1_addrh, dma_tag);
	//wait for all 3 channels to finish
	count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		invalid += tngetd(STS_DMA0_RX);
		invalid += tngetd(STS_DMA1_RX);
	} while (invalid);
	status = getd(STS_DMA0_RX);
	check_DMA_status(status, len, dma_tag1, 0);	
	status = getd(STS_DMA1_RX);
	check_DMA_status(status, len, dma_tag2, 0);
	//TODO: check status of arith ip
	//finally update outbound_seq
	world->ranks[dst_rank].outbound_seq++;
	return COLLECTIVE_OP_SUCCESS;
}

//performs an accumulate using DMA0 and DMA1 and then moves the result 
//through tx_subsystem to dst_rank
static inline int reduce_offchip(	
					unsigned int comm,
					unsigned int dst_rank,
					unsigned int len, 
					unsigned int func,
					unsigned int op0_addrl,
					unsigned int op0_addrh,
					unsigned int op1_addrl,
					unsigned int op1_addrh,
					unsigned int dst_tag) {
	
	//find communicator
	communicator world = find_comm(comm);
	return reduce_offchip_world(&world, dst_rank, len, func, op0_addrl, op0_addrh, op1_addrl, op1_addrh, dst_tag);
}

//copy from mailbox memory into program memory
//this will enable updating the elf without a debug port
//TODO figure out how this can happen without the processor
//overwriting its own program
void update_program(void){
	int i;
	unsigned int val;
	for(i=0; i<16384; i++){
		val = Xil_In32(offsets[0]+4*i);
		Xil_Out32(0x10000+4*i,val);
	}
}

//RX address queue management
//maintaint a list of N buffers in device memory
//queue those buffers for receives
unsigned int enqueue_rx_buffers(void){
	unsigned int ret = 0, cmd_queue;
	unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	int i,new_dma_tag;
	cmd_queue = use_tcp ? CMD_DMA2_TX : CMD_DMA0_TX;
	for(i=0; i<nbufs; i++){
		if(num_rx_enqueued >= 16) return ret; //not enough dma tag
		if(rx_buf_list[i].status   != STATUS_IDLE) continue;
		//found a spare buffer to enqueue 
		//look for a new dma tag
		//TODO:speedup new_dma tag using dma_tag variable
		for(new_dma_tag=0; new_dma_tag < MAX_DMA_TAGS && dma_tag_lookup[new_dma_tag] != -1; new_dma_tag++);
		//new_dma_tag now holds the new dma tag to use
		if( new_dma_tag >= MAX_DMA_TAGS) return ret; //shouldn't happen : since num_rx_enqueued guard against that. but to guard against wrong handling of num_rx_enqueued
		//whatever we find now we can enqueue
		dma_cmd(cmd_queue, MAX_BTT, rx_buf_list[i].addrl, rx_buf_list[i].addrh, new_dma_tag);
		rx_buf_list[i].status 	= STATUS_ENQUEUED;
		rx_buf_list[i].dma_tag 	= new_dma_tag;
		dma_tag_lookup[new_dma_tag] = i;
		//next_rx_tag = (next_rx_tag + 1) & 0xf;
		num_rx_enqueued++;
		ret ++;
	}

	return ret;
}

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

//iterates over rx buffers until match is found or a timeout expires
//matches len, src and tag if tag is not ANY
//returns the index of the spare_buffer
//timeout is jumps to exception handler
int receive_offchip_world_i(	
						communicator *world,	
						unsigned int src_rank,
						unsigned int len,
						unsigned int src_tag
					){
	unsigned int seq_num, src_port, i;
	seq_num 	= world->ranks[src_rank].inbound_seq;
	//src_port	= world->ranks[src_rank].port; TODO: use session to support multiple communicators/multpile connections between same ranks 
	//parse rx buffers until match is found
	//matches len, src
	//matches tag or tag is ANY
	//return buffer index 
	unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	for(unsigned int count = 0; timeout == 0 || count < timeout; count ++){
		for(i=0; i<nbufs; i++){	
			microblaze_disable_interrupts();
			if(rx_buf_list[i].status == STATUS_RESERVED)
			{
				if((rx_buf_list[i].rx_src == src_rank) && (rx_buf_list[i].rx_len == len))
				{
					if(((rx_buf_list[i].rx_tag == src_tag) || (src_tag == TAG_ANY)) && (rx_buf_list[i].sequence_number == seq_num) )
					{
						//only now advance sequence number
						world->ranks[src_rank].inbound_seq++;
						microblaze_enable_interrupts();
						return i;
					}
				}
			}
			microblaze_enable_interrupts();
		}
	}
	longjmp(excp_handler, RECEIVE_TIMEOUT_ERROR);
}

#ifdef OPT_SEG
//Receive data from the rx buffer and accumulate rx buffer with the dst buffer
static inline int accumulate_offchip_world (	
						communicator *world,	
						unsigned int len,
						unsigned int func,
						unsigned int src_rank,
						unsigned int src_tag,
						unsigned int dst_addrl,
						unsigned int dst_addrh) {
	int buf_idx;
	int ret = 0;
	//get rx_buff_location
	unsigned int nbufs 	   	  = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list 	  = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

	int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
	uint64_t curr_addr = (uint64_t)dst_addrl | ( (uint64_t)dst_addrh << 32);
	unsigned int curr_addrl, curr_addrh;
	unsigned int curr_len = MAX_SEG_SIZE;
	unsigned int recv_len = 0;

	for (int i = 0; i < num_segment; ++i)
	{
		if (recv_len + MAX_SEG_SIZE > len)
			curr_len = len - recv_len;
		else
			curr_len = MAX_SEG_SIZE;
		curr_addrl = curr_addr & 0xFFFFFFFF;
		curr_addrh = (curr_addr >> 32) & 0xFFFFFFFF;

		//1. receive part of data and retrieve spare buffer index
		buf_idx = receive_offchip_world_i(world, src_rank, curr_len, src_tag);
		if  (buf_idx < 0 || buf_idx >= nbufs )
			return -1;
		//2. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
		ret += reduce_loopback(curr_len, func, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, curr_addrl, curr_addrh, curr_addrl, curr_addrh);

		//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();

		curr_addr = curr_addr + curr_len;
		recv_len = recv_len + curr_len;
	}
	return ret;
}

#else
//Receive data from the rx buffer and accumulate rx buffer with the dst buffer
static inline int accumulate_offchip_world (	
						communicator *world,	
						unsigned int len,
						unsigned int func,
						unsigned int src_rank,
						unsigned int src_tag,
						unsigned int dst_addrl,
						unsigned int dst_addrh) {
	int buf_idx;
	int ret = 0;
	//get rx_buff_location
	unsigned int nbufs 	   	  = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list 	  = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//1. receive part of data and retrieve spare buffer index
	buf_idx = receive_offchip_world_i(world, src_rank, len, src_tag);
	if  (buf_idx < 0 || buf_idx >= nbufs )
		return -1;
	//2. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
	ret += reduce_loopback(len, func, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, dst_addrl, dst_addrh, dst_addrl, dst_addrh);

	//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
	microblaze_disable_interrupts();
	rx_buf_list[buf_idx].status = STATUS_IDLE;
	microblaze_enable_interrupts();
	
	return ret;
}
#endif

#ifdef OPT_SEG
// int receive_offchip(
// 						communicator *world,	
// 						unsigned int src_rank,	
// 						unsigned int len,
// 						unsigned int dst_addrl,
// 						unsigned int dst_addrh,
// 						unsigned int src_tag
// 						){
// 	unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
// 	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
// 	unsigned int ret;
// 	int buf_idx = receive_offchip_world_i(world, src_rank, len, src_tag);
// 	if  (buf_idx < 0 || buf_idx >= nbufs )
// 		return -1;
	
// 	ret = dma_loopback(len, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, dst_addrl, dst_addrh);
// 	//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
// 	microblaze_disable_interrupts();
// 	rx_buf_list[buf_idx].status = STATUS_IDLE;
// 	microblaze_enable_interrupts();
// 	return ret;
// }

int receive_offchip_world(communicator *world,	
						unsigned int src_rank,	
						unsigned int len,
						unsigned int dst_addrl,
						unsigned int dst_addrh,
						unsigned int src_tag){

	int ret = 0;

	unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

	int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
	uint64_t curr_addr = (uint64_t)dst_addrl | ( (uint64_t)dst_addrh << 32);
	unsigned int curr_addrl, curr_addrh;
	unsigned int curr_len = MAX_SEG_SIZE;
	unsigned int recv_len = 0;

	for (int i = 0; i < num_segment; ++i)
	{
		if (recv_len + MAX_SEG_SIZE > len)
			curr_len = len - recv_len;
		else
			curr_len = MAX_SEG_SIZE;
		curr_addrl = curr_addr & 0xFFFFFFFF;
		curr_addrh = (curr_addr >> 32) & 0xFFFFFFFF;

		int buf_idx = receive_offchip_world_i(world, src_rank, curr_len, src_tag);
		if  (buf_idx < 0 || buf_idx >= nbufs )
			return -1;
		ret += dma_loopback(curr_len, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, curr_addrl, curr_addrh);

		//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();

		// ret += receive_offchip(world, src_rank, curr_len, curr_addrl, curr_addrh, src_tag);

		curr_addr = curr_addr + curr_len;
		recv_len = recv_len + curr_len;
	}
	return ret;
}
#else
//iterates over rx buffers until match is found or a timeout expires
//matches len, src and tag if tag is not ANY
//move data in the supplied memory space
//invalid buffer is indicated by return -1
int receive_offchip_world(
						communicator *world,	
						unsigned int src_rank,	
						unsigned int len,
						unsigned int dst_addrl,
						unsigned int dst_addrh,
						unsigned int src_tag
						){
	unsigned int nbufs = Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	unsigned int ret;
	int buf_idx = receive_offchip_world_i(world, src_rank, len, src_tag);
	if  (buf_idx < 0 || buf_idx >= nbufs )
		return -1;
	
	ret = dma_loopback(len, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, dst_addrl, dst_addrh);
	//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
	microblaze_disable_interrupts();
	rx_buf_list[buf_idx].status = STATUS_IDLE;
	microblaze_enable_interrupts();
	return ret;
}
#endif

int send(		
				unsigned int comm,
				unsigned int len,
				unsigned int tag,
				unsigned int dst_rank,
				unsigned int buf_addrl,
				unsigned int buf_addrh){
	int ret  = 0;
	//send
	ret 	+= transmit_offchip(comm, dst_rank, len, buf_addrl, buf_addrh, tag);
	return ret;
}


int openCon (unsigned int comm
	)
{
	unsigned int session 	= 0;
	unsigned int dst_ip 	= 0;
	unsigned int dst_port 	= 0;
	unsigned int success	= 0;
	int ret = 0;

	unsigned int cur_rank_ip	= 0;
	unsigned int cur_rank_port 	= 0;
	//find communicator
	communicator world 		= find_comm(comm);

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

int openPort(unsigned int comm)
{
	int success = 0;
	//find communicator
	communicator world = find_comm(comm);

	//open port with only the local rank
	putd(CMD_TCP_PORT, world.ranks[world.local_rank].port);
	success = getd(STS_TCP_PORT);

	if (success)
		return COLLECTIVE_OP_SUCCESS;
	else
		return OPEN_PORT_NOT_SUCCEEDED;

}

int init_connection (unsigned int comm)
{
	int ret_openPort 	= openPort(comm);
	int ret_openCon 	= openCon(comm);

	int retval = (ret_openPort | ret_openCon);

	return retval;
}


int recv(		
				unsigned int comm,
				unsigned int len,
				unsigned int tag,
				unsigned int src_rank,
				unsigned int buf_addrl,
				unsigned int buf_addrh){
	int ret = 0;
	//find communicator
	communicator world = find_comm(comm);
	//receive
	ret += receive_offchip_world(&world, src_rank, len, buf_addrl, buf_addrh,  tag);

	return ret;
}

#ifdef OPT_SEG_INTERLEAVE_BCAST

// Interleave segmentations of different nodes
int broadcast(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned int buf_addrl,
				unsigned int buf_addrh){
	int ret = 0;

	int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
	uint64_t curr_addr = (uint64_t)buf_addrl | ( (uint64_t)buf_addrh << 32);
	unsigned int curr_addrl, curr_addrh;
	unsigned int curr_len = MAX_SEG_SIZE;
	unsigned int sent_len = 0;

	//find communicator
	communicator world = find_comm(comm);

	//determine if we're sending or receiving
	if(src_rank == world.local_rank)
	{
		//configure switch
		if (use_tcp){
			cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_INTERNAL);
		}else{
			cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_INTERNAL);
		}

		//Send dma commands
		for (int j = 0; j < num_segment; ++j)
		{
			if (sent_len + MAX_SEG_SIZE > len)
				curr_len = len - sent_len;
			else
				curr_len = MAX_SEG_SIZE;
			curr_addrl = curr_addr & 0xFFFFFFFF;
			curr_addrh = (curr_addr >> 32) & 0xFFFFFFFF;
			//send to all members of the communicator
			for(int i=0; i<world.size; i++)
			{
				if(i != world.local_rank)
				{
					ret += transmit_offchip_cmd(&world, i, curr_len, curr_addrl, curr_addrh, TAG_ANY);
				}
			}
			curr_addr = curr_addr + curr_len;
			sent_len = sent_len + curr_len;
		}

		//Collect dma responses
		for (int j = 0; j < num_segment; ++j)
		{
			for (int i = 0; i < world.size -1; ++i)
			{
				//TODO check status!!
				transmit_offchip_sts();
			}
		}	
		
	}else
	{
		ret += 	receive_offchip_world(	&world, src_rank, len, buf_addrl, buf_addrh, TAG_ANY);
	}
	return ret;
}

#else

int broadcast(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned int buf_addrl,
				unsigned int buf_addrh){
	int i;
	int ret = 0, cmd_queue;
	unsigned int dst_port, sequence_number, dma_tag1;
	unsigned int invalid, status, count = 0;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		//configure switch
		if (use_tcp){
			cmd_queue = CMD_TCP_TX ;
			cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_INTERNAL);
		}else{
			cmd_queue = CMD_UDP_TX;
			cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_INTERNAL);
		}

		//send to all members of the communicator
		dma_tag1 = dma_tag;
		for(i=0; i<world.size; i++){
			if(i == world.local_rank) continue;
			
			//prepare header arguments
			//src_rank 		= world->local_rank;
			sequence_number = world.ranks[i].outbound_seq + 1;
			dst_port  		= use_tcp ? world.ranks[i].session : world.ranks[i].port;
			//send command to packetizer
			putd(cmd_queue, dst_port);
			putd(cmd_queue, len);
			putd(cmd_queue, TAG_ANY);
			putd(cmd_queue, src_rank);
			cputd(cmd_queue,sequence_number);
			//start DMA1
			dma_tag1 = (dma_tag1 + 1) & 0xf;
			dma_cmd(CMD_DMA1_RX, len, buf_addrl, buf_addrh, dma_tag1);
			
		}
		//wait results
		for(i=0; i<world.size; i++){
			if(i== world.local_rank) continue;
			count = 0;
			do {
				if(timeout != 0 && count >= timeout )
					longjmp(excp_handler, DMA_TIMEOUT_ERROR);
				count ++;
				invalid = tngetd(STS_DMA1_RX);
			} while (invalid);
			status = getd(STS_DMA1_RX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, len, dma_tag, 0);	
			//finally update outbound_seq
			world.ranks[i].outbound_seq++;
		} 
	}else{

		ret += 	receive_offchip_world(	&world, src_rank, len, buf_addrl, buf_addrh, TAG_ANY);
	}
	return ret;
}
#endif

int scatter(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned int src_buf_addrl,
				unsigned int src_buf_addrh,
				unsigned int dst_buf_addrl,
				unsigned int dst_buf_addrh){
	int i, cmd_queue;
	int ret = 0;
	unsigned long long tmp_buf_addr;
	unsigned int tmp_buf_addrh, tmp_buf_addrl;
	unsigned int dst_port, sequence_number, dma_tag1;
	unsigned int invalid, status, count = 0;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		tmp_buf_addr  = src_buf_addrh;
		tmp_buf_addr  = tmp_buf_addr <<32 | src_buf_addrl;
		tmp_buf_addrl = src_buf_addrl;
		tmp_buf_addrh = src_buf_addrh;
		//for all non root ranks of the communicator
		//configure switch
		if (use_tcp){
			cmd_queue = CMD_TCP_TX ;
			cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_INTERNAL);
		}else{
			cmd_queue = CMD_UDP_TX;
			cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_INTERNAL);
		}
		//send to all other members of the communicator
		dma_tag1 = dma_tag;
		for(i=0; i<world.size; i++){
			if(i != world.local_rank){
				//prepare header arguments
				//src_rank 		= world->local_rank;
				dst_port 		= use_tcp ? world.ranks[i].session : world.ranks[i].port;
				sequence_number = world.ranks[i].outbound_seq + 1;
				//send command to packetizer
				putd(cmd_queue, dst_port);
				putd(cmd_queue, len);
				putd(cmd_queue, TAG_ANY);
				putd(cmd_queue, src_rank);
				cputd(cmd_queue,sequence_number);
				//start DMA1
				dma_tag1 = (dma_tag1 + 1) & 0xf;
				dma_cmd(CMD_DMA1_RX, len, tmp_buf_addrl, tmp_buf_addrh, dma_tag1);
			}
			tmp_buf_addr += len;
			tmp_buf_addrl = tmp_buf_addr & 0xFFFFFFFF;
			tmp_buf_addrh = (tmp_buf_addr>>32) & 0xFFFFFFFF;
		}
		//wait results
		for(i=0; i<world.size; i++){
			if(i== world.local_rank) continue;
			count =0;
			do {
				if(timeout != 0 && count >= timeout )
					longjmp(excp_handler, DMA_TIMEOUT_ERROR);
				count ++;
				invalid = tngetd(STS_DMA1_RX);
			} while (invalid);
			status = getd(STS_DMA1_RX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, len, dma_tag, 0);	
			//finally update outbound_seq
			world.ranks[i].outbound_seq++;
		} 

		//for root copy
		tmp_buf_addr  = src_buf_addrh;
		tmp_buf_addr  = tmp_buf_addr <<32 | src_buf_addrl;
		tmp_buf_addr += len * world.local_rank;
		tmp_buf_addrl = tmp_buf_addr & 0xFFFFFFFF;
		tmp_buf_addrh = (tmp_buf_addr>>32) & 0xFFFFFFFF;
		ret += dma_loopback(	len, tmp_buf_addrl, tmp_buf_addrh, dst_buf_addrl, dst_buf_addrh);
		
	}else{
		ret += receive_offchip_world(			&world, src_rank, len, dst_buf_addrl, dst_buf_addrh,  TAG_ANY);
	}
	return ret;
}

int gather(		
				unsigned int comm,
				unsigned int len,
				unsigned int root_rank,
				unsigned int src_buf_addrl,
				unsigned int src_buf_addrh,
				unsigned int dst_buf_addrl,
				unsigned int dst_buf_addrh){
	int i;
	int ret = 0;
	unsigned long long dst_buf_addr;
	return COLLECTIVE_NOT_IMPLEMENTED; //at this moment is not safe to run non ring based collectives. they are not handled at depacketizer level.
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(root_rank == world.local_rank){
		//receive from all members of the communicator
		dst_buf_addr  = dst_buf_addrh;
		dst_buf_addr  = dst_buf_addr <<32 | dst_buf_addrl;
		for(i=0; i<world.size; i++){
			//TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
			//currently this will go sequentially through the segments, which is slower and also requires more spare RX buffers
			if(i==world.local_rank)
			{ // root copies
				ret	+= copy(				len, src_buf_addrl, src_buf_addrh, dst_buf_addrl, dst_buf_addrh);
			}else{
				ret += receive_offchip_world(	&world, i		 , len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			}
			dst_buf_addr += len;
			dst_buf_addrl = dst_buf_addr & 0xFFFFFFFF;
			dst_buf_addrh = (dst_buf_addr>>32) & 0xFFFFFFFF;
		}
	}else{
		ret += 		transmit_offchip_world(		&world, root_rank, len, src_buf_addrl, src_buf_addrh, TAG_ANY);
	}
	return ret;
}

int gather_ring(
				unsigned int comm,
				unsigned int len,
				unsigned int root_rank,
				unsigned int src_buf_addrl,
				unsigned int src_buf_addrh,
				unsigned int dst_buf_addrl,
				unsigned int dst_buf_addrh){
	int ret = 0;
	unsigned long long dst_buf_addr, tmp_buf_addr;
	unsigned int i,curr_pos, next_in_ring, prev_in_ring, number_of_shift, buf_idx;
	//find communicator
	communicator world = find_comm(comm);
	next_in_ring = (world.local_rank+1			   ) % world.size	;
	prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	
	if(root_rank == world.local_rank){ //root ranks mainly receives
		dst_buf_addr  = dst_buf_addrh;
		dst_buf_addr  = dst_buf_addr <<32 | dst_buf_addrl;
		//receive from all members of the communicator
		for(i=0, curr_pos = prev_in_ring; i<world.size; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
			//TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
			//currently this will go sequentially through the segments, which is slower and also requires more spare RX buffers
			tmp_buf_addr  = dst_buf_addr + len*curr_pos;
			dst_buf_addrl = tmp_buf_addr & 0xFFFFFFFF;
			dst_buf_addrh = (tmp_buf_addr>>32) & 0xFFFFFFFF;
			if(curr_pos==world.local_rank)
			{ // root copies
				ret	+= copy(	     len, src_buf_addrl, src_buf_addrh, dst_buf_addrl, dst_buf_addrh);
			}else{
				ret += receive_offchip_world(&world, prev_in_ring, len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			}
		}
	}else{
		//non root ranks sends their data + relay others data to the next rank in sequence
		// as a daisy chain
		unsigned int nbufs 	   = 	 Xil_In32(RX_BUFFER_COUNT_OFFSET);
		rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
		number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ; //distance to the root
		ret += transmit_offchip_world(		&world, next_in_ring, len, src_buf_addrl, src_buf_addrh, TAG_ANY);
		for (int i = 0; i < number_of_shift; i++)
		{	//relay the others 
			#ifdef OPT_SEG
			// ret += receive_offchip_world( 	 &world, prev_in_ring, len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			// ret += transmit_offchip_world(	 &world, next_in_ring, len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
			unsigned int curr_len = MAX_SEG_SIZE;
			unsigned int sent_len = 0;
			for (int j = 0; j < num_segment; ++j)
			{
				if (sent_len + MAX_SEG_SIZE > len)
					curr_len = len - sent_len;
				else
					curr_len = MAX_SEG_SIZE;
				buf_idx = receive_offchip_world_i (&world, prev_in_ring, curr_len, TAG_ANY);
				if(buf_idx < 0) return -1;
				ret += transmit_offchip_world(&world, next_in_ring, curr_len, rx_buf_list[i].addrl, rx_buf_list[i].addrh, TAG_ANY);
				sent_len = sent_len + curr_len;
				//housekeeping: release spare buffer
				microblaze_disable_interrupts();
				rx_buf_list[buf_idx].status = STATUS_IDLE;
				microblaze_enable_interrupts();
			}
			#else
			//relay the others v2
			buf_idx    = receive_offchip_world_i(  &world, prev_in_ring, len, TAG_ANY);
			if(buf_idx < 0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
			ret += transmit_offchip_world(	 &world, next_in_ring, len, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, TAG_ANY);
			//housekeeping: release spare buffer
			microblaze_disable_interrupts();
			rx_buf_list[buf_idx].status = STATUS_IDLE;
			microblaze_enable_interrupts();
			#endif
		}	
	}
	return ret;
}

static inline int allgather(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){	
	int ret = 0;
	//find communicator
	communicator world = find_comm(comm);
	//let 0 be the root
	int root = 0; 
	ret +=gather_ring(	comm, len			, root, src_addrl, src_addrh, dst_addrl, dst_addrh);
	ret +=broadcast(	comm, len*world.size, root, dst_addrl, dst_addrh);
	return ret;
}

int allgather_ring(
				unsigned int comm,
				unsigned int len,
				unsigned int src_buf_addrl,
				unsigned int src_buf_addrh,
				unsigned int dst_buf_addrl,
				unsigned int dst_buf_addrh){
	int ret = 0;
	unsigned long long dst_buf_addr, tmp_buf_addr;
	unsigned int i,curr_pos, next_in_ring, prev_in_ring;
	//find communicator
	communicator world = find_comm(comm);
	next_in_ring = (world.local_rank+1			   ) % world.size	;
	prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	//load buf_addr
	dst_buf_addr  = dst_buf_addrh;
	dst_buf_addr  = dst_buf_addr <<32 | dst_buf_addrl;
	//send our data to next in ring
	ret += transmit_offchip_world(				&world, next_in_ring,len, src_buf_addrl	, src_buf_addrh, TAG_ANY);
	//receive from all members of the communicator
	for(i=0, curr_pos = world.local_rank; i<world.size; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
		tmp_buf_addr  = dst_buf_addr + len*curr_pos;
		dst_buf_addrl = tmp_buf_addr & 0xFFFFFFFF;
		dst_buf_addrh = (tmp_buf_addr>>32) & 0xFFFFFFFF;

		if(curr_pos==world.local_rank){
			ret	+= copy(len, src_buf_addrl, src_buf_addrh, dst_buf_addrl, dst_buf_addrh);
		}else{
			ret += receive_offchip_world(	  	&world, prev_in_ring, len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			//TODO: use the same stream to move data both to dst_buf_addr and tx_subsystem
			if(i+1 < world.size){ //if not the last data needed relay to the next in the sequence
				ret += transmit_offchip_world(	&world, next_in_ring, len, dst_buf_addrl, dst_buf_addrh, TAG_ANY);
			}
		}		
	}

	return ret;
}

int reduce(		
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int root_rank,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){
	return COLLECTIVE_NOT_IMPLEMENTED; //at this moment is not safe to run non ring based collectives. they are not handled at depacketizer level.
	int ret = 0;
	// int buf_idx;
	//find communicator
	communicator world = find_comm(comm);
	//get rx_buff_location //TODO: write a function for that or move it into a static variable
	// unsigned int nbufs 	   = 	 Xil_In32(RX_BUFFER_COUNT_OFFSET);
	// rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//determine if we're sending or receiving
	if(root_rank == world.local_rank){
		//0. copy src_buffer of master in dst_buffer
		// from now on dst_buffer will represent the accumulator
		ret += copy(len, src_addrl, src_addrh, dst_addrl, dst_addrh);
		//receive and accumulate from all members of the communicator
		for(int i=0; i<world.size; i++){
			//0. skip if we are root rank
			if(i == root_rank)
				continue;

			//1. receive part of data and retrieve spare buffer index
			buf_idx = receive_offchip_world_i(&world, i, len, TAG_ANY);
			if  (buf_idx < 0 || buf_idx >= nbufs )
				return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
			//2. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
			//configure switch
			ret += accumulate(len, function, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, dst_addrl, dst_addrh,  dst_addrl, dst_addrh);
			//3. release buffer
			microblaze_disable_interrupts();
			rx_buf_list[buf_idx].status = STATUS_IDLE;
			microblaze_enable_interrupts();
			//move to next rank
		}
	}else{
		ret += transmit_offchip_world(&world, root_rank, len, src_addrl, src_addrh,  TAG_ANY);
	}
	return ret;
}


int reduce_ring(
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int root_rank,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){
	int ret = COLLECTIVE_OP_SUCCESS;
	int buf_idx;

	//find communicator
	communicator world 		  = find_comm(comm);
	unsigned int next_in_ring = (world.local_rank+1			   ) % world.size	;
	unsigned int prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	//get rx_buff_location
	unsigned int nbufs 	   	  = 	  Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list 	  = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//determine if we're sending or receiving
	if(root_rank == world.local_rank){
		ret += accumulate_offchip_world(&world, len, function, prev_in_ring, TAG_ANY, dst_addrl, dst_addrh); 
	}else if( prev_in_ring == root_rank){ 
		//non root ranks immediately after the root sends
		ret = transmit_offchip_world(&world, next_in_ring, len, src_addrl, src_addrh, TAG_ANY);
	}else{
		//non root ranks sends their data + data received from previous rank to the next rank in sequence
		// as a daisy chain
		buf_idx = receive_offchip_world_i (&world, prev_in_ring, len, TAG_ANY);
		if(buf_idx < 0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		ret += reduce_offchip_world(&world, next_in_ring, len, function, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, src_addrl, src_addrh,  TAG_ANY);
		//housekeeping: release spare buffer
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();
	}

	//	unsigned int number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ;
	//	ret += transmit_offchip_world(&world, next_in_ring,	len, src_addrl, src_addrh, TAG_ANY );
	//	for (int i = 0; i < number_of_shift; i++)
	//	{	
	//		#ifdef OPT_SEG
	//		// ret += receive_offchip_world( 	 &world, prev_in_ring, len, dst_addrl, dst_addrh, TAG_ANY);
	//		// ret += transmit_offchip_world(	 &world, next_in_ring, len, dst_addrl, dst_addrh, TAG_ANY);
	//		int num_segment = (len + MAX_SEG_SIZE - 1) / MAX_SEG_SIZE;
	//		unsigned int curr_len = MAX_SEG_SIZE;
	//		unsigned int sent_len = 0;
	//		for (int j = 0; j < num_segment; ++j)
	//		{
	//			if (sent_len + MAX_SEG_SIZE > len)
	//				curr_len = len - sent_len;
	//			else
	//				curr_len = MAX_SEG_SIZE;
	//			buf_idx = receive_offchip_world_i (&world, prev_in_ring, curr_len, TAG_ANY);
	//			if(buf_idx < 0) return -1;
	//			ret += transmit_offchip_world(&world, next_in_ring, curr_len, rx_buf_list[i].addrl, rx_buf_list[i].addrh, TAG_ANY);
	//			sent_len = sent_len + curr_len;
	//			//housekeeping: release spare buffer
	//			microblaze_disable_interrupts();
	//			rx_buf_list[buf_idx].status = STATUS_IDLE;
	//			microblaze_enable_interrupts();
	//		}
	//		#else
	//		buf_idx = receive_offchip_world_i (&world, prev_in_ring, len, TAG_ANY);
	//		if(buf_idx < 0) return -1;
	//		ret += transmit_offchip_world(&world, next_in_ring, len, rx_buf_list[i].addrl, rx_buf_list[i].addrh, TAG_ANY);
	//		//housekeeping: release spare buffer
	//		microblaze_disable_interrupts();
	//		rx_buf_list[buf_idx].status = STATUS_IDLE;
	//		microblaze_enable_interrupts();
	//		#endif
	//	}

	
	return ret;
}

int allreduce(	
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){

	int ret = 0;
	//let 0 be the root
	int root = 1; 
	reduce_ring(comm, len, function, root, src_addrl, src_addrh, dst_addrl, dst_addrh);
	broadcast(	comm, len,			 root, dst_addrl, dst_addrh);
	return 0;
}

#ifdef OPT_RING_ALL_REDUCE

int allreduce_fused_ring(	
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){

	int ret = 0;
	//find communicator
	communicator world 		  	= find_comm(comm);
	unsigned int next_in_ring 	= (world.local_rank+1			   )%world.size	;
	unsigned int prev_in_ring 	= (world.local_rank+world.size-1 )%world.size	;
	//get rx_buff_location
	// unsigned int nbufs 	   	  	=     Xil_In32(RX_BUFFER_COUNT_OFFSET);
	// rx_buffer *rx_buf_list 	  	= (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	// unsigned int buf_idx;

	unsigned int len_div_size = len/world.size; 
	unsigned int curr_send_chunk = world.local_rank;
	unsigned int curr_recv_chunk = (curr_send_chunk + world.size - 1) % world.size;

	uint64_t src_addr = (uint64_t)src_addrl | ( (uint64_t)src_addrh << 32);
	uint64_t curr_src_addr;
	unsigned int curr_src_addrl, curr_src_addrh;

	uint64_t curr_accu_addr;
	unsigned int curr_accu_addrl, curr_accu_addrh;

	uint64_t dst_addr = (uint64_t)dst_addrl | ( (uint64_t)dst_addrh << 32);
	uint64_t curr_dst_addr;
	unsigned int curr_dst_addrl, curr_dst_addrh;

	//Share reduce stage
	for (int i = 0; i < world.size -1; ++i)
	{
		curr_src_addr = src_addr + len_div_size * curr_send_chunk;
		curr_src_addrl = curr_src_addr & 0xFFFFFFFF;
		curr_src_addrh = (curr_src_addr >> 32) & 0xFFFFFFFF;

		curr_accu_addr = src_addr + len_div_size * curr_recv_chunk;
		curr_accu_addrl = curr_accu_addr & 0xFFFFFFFF;
		curr_accu_addrh = (curr_accu_addr >> 32) & 0xFFFFFFFF;

		// TODO: 1 and 3 can actually overlapped
		// 1.send our data from src buffer
		ret += transmit_offchip_world(&world, next_in_ring,	len_div_size, curr_src_addrl, curr_src_addrh, TAG_ANY);
		//2. receive part of data from previous in rank and retrieve spare buffer index
		//3. accumulate in src buffer: src_buffer = src_buffer + spare_buffer 
		ret += accumulate_offchip_world(&world, len_div_size, function, prev_in_ring, TAG_ANY, curr_accu_addrl, curr_accu_addrh); 

		// buf_idx = receive_offchip_world_i(&world, prev_in_ring,len_div_size, TAG_ANY);
		// if  (buf_idx < 0 || buf_idx >= nbufs )
		// 	return -1;
		// ret += accumulate(len_div_size, function , rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, curr_accu_addrl, curr_accu_addrh);

		//4. if last iteration, copy the reduced result from src buffer to corresponding position of the dst buffer
		if (i == world.size - 2)
		{
			curr_dst_addr = dst_addr + len_div_size * curr_recv_chunk;
			curr_dst_addrl = curr_dst_addr & 0xFFFFFFFF;
			curr_dst_addrh = (curr_dst_addr >> 32) & 0xFFFFFFFF;
			ret += copy(len_div_size, curr_accu_addrl, curr_accu_addrh, curr_dst_addrl, curr_dst_addrh);
		}

		curr_send_chunk = curr_recv_chunk;
		curr_recv_chunk = (curr_recv_chunk + world.size - 1) % world.size;

		//housekeeping: release spare buffer
		// microblaze_disable_interrupts();
		// rx_buf_list[buf_idx].status = STATUS_IDLE;
		// microblaze_enable_interrupts();
	}

	//Share result stage
	for (int i = 0; i < world.size -1; ++i)
	{
		//5. send the local reduced results to next rank
		curr_dst_addr = dst_addr + len_div_size * curr_send_chunk;
		curr_dst_addrl = curr_dst_addr & 0xFFFFFFFF;
		curr_dst_addrh = (curr_dst_addr >> 32) & 0xFFFFFFFF;
		ret += transmit_offchip_world(&world, next_in_ring,	len_div_size, curr_dst_addrl, curr_dst_addrh, TAG_ANY);

		//6. receive the reduced results from previous rank and put into correct dst buffer position
		curr_dst_addr = dst_addr + len_div_size * curr_recv_chunk;
		curr_dst_addrl = curr_dst_addr & 0xFFFFFFFF;
		curr_dst_addrh = (curr_dst_addr >> 32) & 0xFFFFFFFF;

		ret += receive_offchip_world(&world, prev_in_ring,	len_div_size, curr_dst_addrl, curr_dst_addrh, TAG_ANY);

		curr_send_chunk = curr_recv_chunk;
		curr_recv_chunk = (curr_recv_chunk + world.size - 1) % world.size;

		//housekeeping: release spare buffer
		// microblaze_disable_interrupts();
		// rx_buf_list[buf_idx].status = STATUS_IDLE;
		// microblaze_enable_interrupts();

	}

	return ret;
}

#else

int allreduce_fused_ring(	
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int src_addrl,
				unsigned int src_addrh,
				unsigned int dst_addrl,
				unsigned int dst_addrh){

	int ret = COLLECTIVE_OP_SUCCESS;
	//find communicator
	communicator world 		  	= find_comm(comm);
	unsigned int next_in_ring 	= (world.local_rank+1			   )%world.size	;
	unsigned int prev_in_ring 	= (world.local_rank+world.size-1 )%world.size	;
	//get rx_buff_location
	unsigned int nbufs 	   	  	=     Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list 	  	= (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	unsigned int buf_idx;
	//every member of the communicator 
	// 0.  sends its data to the next rank
	// 1. put their own data in their buffer
	// then receive others data:
	// 2.  receives data from previous rank
	// 3.  relay data to the next in sequence
	// 4.  accumulates data in the destination buffer
	// 0.send our data
	ret += copy(len, src_addrl, src_addrh, dst_addrl, dst_addrh);
	if (ret<0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;

	//1. copy src_buffer of master in dst_buffer
	// from now on dst_buffer will represent the accumulator
	ret += transmit_offchip_world(&world, next_in_ring,			len, src_addrl				  , src_addrh					, TAG_ANY);
	if (ret<0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;

	for(int i=0; i < world.size-1; i++){
		//2. receive part of data from previous in rank and retrieve spare buffer index
		//3. accumulate in buffer: user_buffer = user_buffer + spare_buffer
		// ret += accumulate_offchip_world(&world, len, function, prev_in_ring, TAG_ANY, dst_addrl, dst_addrh); 

		buf_idx = receive_offchip_world_i(&world, prev_in_ring,len, TAG_ANY);
		if  (buf_idx < 0 || buf_idx >= nbufs )
			return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		//4. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
		ret += accumulate(   len, function , rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, dst_addrl, dst_addrh, dst_addrl, dst_addrh);
		if (ret<0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		//3. relay others data only if it's not the last iteration.  
		if (i+1<world.size-1){
			ret += transmit_offchip_world(&world, next_in_ring, len, rx_buf_list[buf_idx].addrl, rx_buf_list[buf_idx].addrh, TAG_ANY);
			if (ret<0) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		}
		//housekeeping: release spare buffer
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();
		//TODO: use reduce_offchip when using tree based collectives
	}
	return ret;
}
#endif

int ext_kernel_stream(
						unsigned int len,
						unsigned int src_addrl,
						unsigned int src_addrh,
						unsigned int dst_addrl,
						unsigned int dst_addrh
){
	int dma_tag1, dma_tag3; 
	//configure switch
	cfg_switch(DATAPATH_DMA_EXT_LOOPBACK, ARITH_INTERNAL);
	//start both channels of the DMA1
	dma_tag1 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA0_RX, len, src_addrl, src_addrh, dma_tag);
	dma_tag3 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA1_TX, len, dst_addrl, dst_addrh, dma_tag);
	//wait for all 3 channels to finish
	unsigned int invalid, status;
	unsigned int count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		invalid += tngetd(STS_DMA0_RX);
		invalid += tngetd(STS_DMA1_TX);
	} while (invalid);
	status = getd(STS_DMA0_RX);
	check_DMA_status(status, len, dma_tag1, 0);
	status = getd(STS_DMA1_TX);
	check_DMA_status(status, len, dma_tag3, 1);
	//BUG: loopback number of
	return COLLECTIVE_OP_SUCCESS;
}

int reduce_ext (	unsigned int len,
					unsigned int op1_addrl,
					unsigned int op1_addrh,
					unsigned int op2_addrl,
					unsigned int op2_addrh,
					unsigned int dst_addrl,
					unsigned int dst_addrh) {
	int dma_tag1,dma_tag2, dma_tag3; 
	//configure switch
	cfg_switch(DATAPATH_DMA_REDUCTION, ARITH_EXTERNAL);
	//start both channels of the DMA1
	dma_tag1 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA0_RX, len, op1_addrl, op1_addrh, dma_tag);
	dma_tag2 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA1_RX, len, op2_addrl, op2_addrh, dma_tag);
	dma_tag3 = ( dma_tag = (dma_tag + 1) & 0xf);
	dma_cmd(CMD_DMA1_TX, len, dst_addrl, dst_addrh, dma_tag);
	//wait for all 3 channels to finish
	unsigned int invalid, status;
	unsigned int count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		invalid += tngetd(STS_DMA0_RX);
		invalid += tngetd(STS_DMA1_RX);
		invalid += tngetd(STS_DMA1_TX);
	} while (invalid);
	status = getd(STS_DMA0_RX);
	check_DMA_status(status, len, dma_tag1, 0);
	status = getd(STS_DMA1_RX);
	check_DMA_status(status, len, dma_tag2, 0);
	status = getd(STS_DMA1_TX);
	check_DMA_status(status, len, dma_tag3, 1);
	
	return COLLECTIVE_OP_SUCCESS;
}


int main() {
	unsigned int retval;
	unsigned int scenario, len, comm, root_src_dst, function, msg_tag;
	unsigned int buf0_type, buf1_type, buf2_type;
	unsigned int buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh, buf2_addrl, buf2_addrh;;

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
		scenario     = getd(CMD_HOST);
		len          = getd(CMD_HOST);
		comm         = getd(CMD_HOST);
		root_src_dst = getd(CMD_HOST);
		function     = getd(CMD_HOST);
		msg_tag      = getd(CMD_HOST);
		buf0_type    = getd(CMD_HOST);
		buf1_type    = getd(CMD_HOST);
		buf2_type    = getd(CMD_HOST);
		buf0_addrl   = getd(CMD_HOST);
		buf0_addrh   = getd(CMD_HOST);
		buf1_addrl   = getd(CMD_HOST);
		buf1_addrh   = getd(CMD_HOST);
		buf2_addrl   = getd(CMD_HOST);
		buf2_addrh   = getd(CMD_HOST);
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
						retval = init_connection(comm);
						break;
					case OPEN_PORT:
						retval = openPort(comm);
						break;
					case OPEN_CON:
						retval = openCon(comm);
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
					default:
						break;
				}
				
				break;
			case XCCL_SEND:
				retval = send(					comm, len, msg_tag,  root_src_dst, 	buf0_addrl, buf0_addrh);
				break;
			case XCCL_RECV:
				retval = recv(					comm, len, msg_tag,  root_src_dst, 	buf0_addrl, buf0_addrh);
				break;
			case XCCL_BCAST:
				retval = broadcast(				comm, len, 			 root_src_dst, 	buf0_addrl, buf0_addrh);
				break;
			case XCCL_SCATTER:	
				retval = scatter(				comm, len, 			 root_src_dst, 	buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_GATHER:
				retval = gather(				comm, len, 			 root_src_dst, 	buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_REDUCE:
				retval = reduce(				comm, len, function, root_src_dst, 	buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_ALLGATHER:
				retval = allgather(				comm, len, 							buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_ALLREDUCE:
				retval = allreduce(				comm, len, function, 				buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_ACC:
				retval = accumulate( 	  	  	  	  len, function, 				buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh,  buf1_addrl, buf1_addrh);
				break;
			case XCCL_COPY:
				retval = copy(	  		  	  	  	  len, 							buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_REDUCE_RING:
				retval = reduce_ring(  			comm, len, function, root_src_dst, 	buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_ALLREDUCE_FUSED_RING:
				retval = allreduce_fused_ring(	comm, len, function, 				buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_GATHER_RING:
				retval = gather_ring(			comm, len, 			 root_src_dst, 	buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_ALLGATHER_RING:
				retval = allgather_ring(		comm, len, 							buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_EXT_STREAM_KRNL:
				retval = ext_kernel_stream( 		  len, 			 				buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh);
				break;
			case XCCL_EXT_REDUCE:
				retval = reduce_ext(				  len, 			 				buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh, buf2_addrl, buf2_addrh);
				break;
			default:
				retval = 0;
				break;
		}

		finalize_call(retval);
	}
	return 0;
}
