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
static volatile	unsigned int dma_transaction_size = DMA_MAX_BTT;


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

//for hls ip protocols https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/managing_interface_synthesis.html#qls1539734256651__ae476333
void start_arith(unsigned int byte_count, unsigned int function, unsigned use_repeat) {
	//get number of 512b/64B transfers corresponding to byte_count
	unsigned int num_dma_transfers 			= (unsigned int)( byte_count / dma_transaction_size);
	unsigned int bytes_tail 				= byte_count % dma_transaction_size;
	unsigned int quad_word_per_dma_transfer = (unsigned int)((dma_transaction_size+63)/64);
	unsigned int quad_word_per_tail			= (unsigned int)((bytes_tail+63)/64);
	//limit the maximum number of transfers since at most we are issueing DMA_MAX_TRANSACTIONS.
	if (num_dma_transfers + ( bytes_tail > 0 ? 1:0)> DMA_MAX_TRANSACTIONS){
		num_dma_transfers = DMA_MAX_TRANSACTIONS;
		quad_word_per_tail = 0;
	}

	unsigned int quad_word_stream_transfers = num_dma_transfers * quad_word_per_dma_transfer + quad_word_per_tail;
	Xil_Out32(ARITH_BASEADDR+0x10, quad_word_stream_transfers);
	Xil_Out32(ARITH_BASEADDR+0x18, function);
	SET(ARITH_BASEADDR, CONTROL_START_MASK | (use_repeat == 1 ? CONTROL_REPEAT_MASK : 0) );
}

static inline void stop_arith() {
	CLR(ARITH_BASEADDR, CONTROL_START_MASK | CONTROL_REPEAT_MASK);
}

static inline int arith_is_idle() {
	return Xil_In32(ARITH_BASEADDR) & CONTROL_IDLE_MASK ;
}

static inline int arith_is_ready() {
	return Xil_In32(ARITH_BASEADDR) & CONTROL_START_MASK ;
}

static inline void arith_disable_auto_repeat() {
	CLR(ARITH_BASEADDR, CONTROL_REPEAT_MASK);
}

static inline void wait_arith(){
	//wait for arithmetich unit to finish
	for(unsigned int count=0; !arith_is_idle() ; count++){
		if(count > timeout)
			longjmp(excp_handler, ARITH_ERROR);
	}
}

static inline void start_packetizer(long long base_addr,unsigned int max_pktsize) {
	//get number of 512b/64B transfers corresponding to max_pktsize
	unsigned int max_pkt_transfers = (max_pktsize+63)/64;
	Xil_Out32(	base_addr +0x10	, max_pkt_transfers);
	SET(		base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK);
}

static inline void start_depacketizer(long long base_addr) {
	SET(		base_addr, CONTROL_REPEAT_MASK | CONTROL_START_MASK );
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

	start_timer1();
	n_enqueued = enqueue_rx_buffers();
	Xil_Out32(TIME_TO_ACCESS_EXCH_MEM, read_timer1());
	
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

//issue a dma command 
static inline void dma_cmd_addrh_addrl(unsigned int channel, unsigned int btt, unsigned int addrh, unsigned int addrl, unsigned int tag){
	putd(channel, 0xC0800000 | btt); // 31=DRR 30=EOF 29-24=DSA 23=Type 22-0=BTT
	putd(channel, addrl);
	putd(channel, addrh);
	cputd(channel, 0x2000 | tag); 	 // 15-12=xCACHE 11-8=xUSER 7-4=RSVD 3-0=TAG
}

//start a DMA operations on a specific channel
static inline void dma_cmd(unsigned int channel, unsigned int btt, unsigned long long addr, unsigned int tag) {
	unsigned int addrl = addr & 0xFFFFFFFF;
	unsigned int addrh = (addr >> 32) & 0xFFFFFFFF;

	dma_cmd_addrh_addrl( channel,  btt, addrh, addrl, tag);
}

static inline void dma_cmd_without_EOF(unsigned int channel, unsigned int btt, unsigned long long addr, unsigned int tag) {
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
//1 == DATAPATH_DMA_LOOPBACK	 : DMA1 RX -> DMA1 TX   (DMA Loopback)
//2 == DATAPATH_DMA_REDUCTION	 : DMA0 RX -> Arith Op0 (DMA Reduction)
//   							   DMA1 RX -> Arith Op1 
//   							   Arith Res -> DMA1 TX 
//3 == DATAPATH_OFFCHIP_TX_UDP   : DMA1 RX -> UDP TX (UDP packetizer)    (Direct Off-chip TX)
//4 == DATAPATH_OFFCHIP_TX_TCP   : DMA1 RX -> TCP TX (TCP packetizer)    (Direct Off-chip TX)
//5 == DATAPATH_OFFCHIP_UDP_REDUCTION: 	DMA0 RX -> Arith Op0 (Reduced Off-chip TX)
//   									DMA1 RX -> Arith Op1 
//   									Arith Res -> UDP TX(UDP packetizer)
//6 == DATAPATH_OFFCHIP_TCP_REDUCTION: 	DMA0 RX -> Arith Op0 (Reduced Off-chip TX)
//   									DMA1 RX -> Arith Op1 
//   									Arith Res -> TCP TX (TCP packetizer)
//7 == DATAPATH_DMA_EXT_LOOPBACK:		DMA0_RX -> M_EXT_KRNL (TX)
//										S_EXT_KRNL (RX) -> DMA1_TX
//there's a subswitch that is specific for reduce:
// you can select where to redirect operands to internal or external arithmetic unit 
//									result from  "             "          "       " 
// ARITH_NONE leaves the current arithmetic switch configuration 
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
		case ARITH_NONE:
			return;
		default:
			break;
	}
	apply_switch_config(ARITH_SWITCH_BASEADDR);
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

//create the instructions for packetizer to send a message. Those infos include, among the others the message header.
unsigned int start_packetizer_message(communicator* world, unsigned int dst_rank, unsigned int len, unsigned int tag){
	unsigned int src_rank, sequence_number, dst, net_tx, curr_len, i;
	unsigned int transmitted = 0; 
	//prepare header arguments
	//TODO: how long does it take to access world (remind that world is in exchange memory)?
	src_rank 		= world->local_rank;
	sequence_number = world->ranks[dst_rank].outbound_seq;
	if (use_tcp){
		dst 			= world->ranks[dst_rank].session; 
		net_tx 			= CMD_TCP_TX; 
	}else{
		dst		 		= world->ranks[dst_rank].port; 
		net_tx		 	= CMD_UDP_TX; 
	}
	//enqueue message headers
	for(i=0; len > 0 && i < DMA_MAX_TRANSACTIONS; i++){
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
static inline void ack_packetizer(communicator * world, unsigned int dst_rank,unsigned int num_msg){
	unsigned int count, ack_seq_num;
	unsigned int packetizer_sts_stream = use_tcp ? STS_TCP_PKT : STS_UDP_PKT;
	while(num_msg > 0){

		for(count=0; tngetd(packetizer_sts_stream); count++){
			if(count > timeout){
				longjmp(excp_handler, PACK_TIMEOUT_STS_ERROR);
			}
		}
		ack_seq_num = getd(packetizer_sts_stream);
		if(	world->ranks[dst_rank].outbound_seq+1 == ack_seq_num){
			world->ranks[dst_rank].outbound_seq+=1;
		} else{
			longjmp(excp_handler, PACK_SEQ_NUMBER_ERROR);
		}
		num_msg -=1;
	}
}

//establish connection with every other rank in the communicator
int openCon (unsigned int comm)
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
//function that was supposed to openPort AND to openCon. 
//Since at this level we are not sure the other ccl_offlaod instances have open their port 
//this function is not working.
int init_connection (unsigned int comm)
{
	int ret_openPort 	= openPort(comm);
	int ret_openCon 	= openCon(comm);

	int retval = (ret_openPort | ret_openCon);

	return retval;
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

static inline unsigned int segment(unsigned int number_of_bytes,unsigned int segment_size){
	return  (number_of_bytes + segment_size - 1) / segment_size;
} 


int dma_movement_single( 	
					unsigned int len,
					unsigned long long DMA0_rx_addr,
					unsigned long long DMA1_rx_addr,
					unsigned long long DMA1_tx_addr,
					unsigned long long DMA2_rx_addr,
					unsigned int what_DMAS) {
	unsigned int invalid, count, status, dma_tag_tmp;
	if(len > DMA_MAX_BTT) return DMA_TRANSACTION_SIZE;
	dma_tag_tmp = dma_tag;
	//start DMAs 
	if( what_DMAS & USE_DMA0_RX){
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
		dma_cmd(CMD_DMA0_RX, len, DMA0_rx_addr, dma_tag_tmp);

	}
	if( what_DMAS & USE_DMA1_RX){
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
		dma_cmd(CMD_DMA1_RX, len, DMA1_rx_addr, dma_tag_tmp);

	}
	if( what_DMAS & USE_DMA1_TX){
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
		dma_cmd(CMD_DMA1_TX, len, DMA1_tx_addr, dma_tag_tmp);

	}
	if( what_DMAS & USE_DMA1_TX_WITHOUT_TLAST){
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
		dma_cmd_without_EOF(CMD_DMA1_TX, len, DMA1_tx_addr, dma_tag_tmp);

	}
	if( what_DMAS & USE_DMA2_RX){
		dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
		dma_cmd(CMD_DMA2_RX, len, DMA2_rx_addr, dma_tag_tmp);

	}
			
	//wait for DMAs to complete
	count = 0;
	do {
		if(timeout != 0 && count >= timeout )
			longjmp(excp_handler, DMA_TIMEOUT_ERROR);
		count ++;
		invalid = 0;
		if( what_DMAS & USE_DMA0_RX){
			invalid += tngetd(STS_DMA0_RX);
		}
		if( what_DMAS & USE_DMA1_RX){
			invalid += tngetd(STS_DMA1_RX);
		}
		if( what_DMAS & (USE_DMA1_TX | USE_DMA1_TX_WITHOUT_TLAST)){
			invalid += tngetd(STS_DMA1_TX);
		}
		if( what_DMAS & USE_DMA2_RX){
			invalid += tngetd(STS_DMA2_RX);
		}
	} while (invalid);

	if( what_DMAS & USE_DMA0_RX){
		status = getd(STS_DMA0_RX);
		dma_tag = (dma_tag + 1) & 0xf;
		check_DMA_status(status, len, dma_tag, 0);
	}
	if( what_DMAS & USE_DMA1_RX){
		status = getd(STS_DMA1_RX);
		dma_tag = (dma_tag + 1) & 0xf;
		check_DMA_status(status, len, dma_tag, 0);
	}
	if( what_DMAS & USE_DMA1_TX){
		status = getd(STS_DMA1_TX);
		dma_tag = (dma_tag + 1) & 0xf;
		check_DMA_status(status, len, dma_tag, 1);
	}
	if( what_DMAS & USE_DMA1_TX_WITHOUT_TLAST){
		status = getd(STS_DMA1_TX);
		dma_tag = (dma_tag + 1) & 0xf;
		check_DMA_status(status, len, dma_tag, 0);
	}
	if( what_DMAS & USE_DMA2_RX){
		status = getd(STS_DMA2_RX);
		dma_tag = (dma_tag + 1) & 0xf;
		check_DMA_status(status, len, dma_tag, 0);
	}

	return COLLECTIVE_OP_SUCCESS;
}

int dma_movement_DMA_MAX_TRANSACTIONS(
	unsigned int len,
	unsigned long long DMA0_rx_addr,
	unsigned long long DMA1_rx_addr,
	unsigned long long DMA1_tx_addr,
	unsigned long long DMA2_rx_addr,
	unsigned int what_DMAS
){
	unsigned int remaining_to_move, remaining_to_ack, moved, dma_tag_tmp, curr_len, invalid, count, status, i;
	moved 				= 0;
	remaining_to_move 	= len;
	remaining_to_ack 	= len;
	dma_tag_tmp 		= dma_tag;

	//issue at most DMA_MAX_TRANSACTIONS of dma_transaction_size
	//start transactions
	for (i = 0; remaining_to_move > 0 && i < DMA_MAX_TRANSACTIONS ; i++){
		if (remaining_to_move > dma_transaction_size)
			curr_len = dma_transaction_size;
		else
			curr_len = remaining_to_move ;
		//start DMAs
		if( what_DMAS & USE_DMA0_RX){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd(CMD_DMA0_RX, curr_len, DMA0_rx_addr, dma_tag_tmp);
			DMA0_rx_addr += curr_len;
		}
		if( what_DMAS & USE_DMA1_RX){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd(CMD_DMA1_RX, curr_len, DMA1_rx_addr, dma_tag_tmp);
			DMA1_rx_addr += curr_len;
		}
		if( what_DMAS & USE_DMA1_TX){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd(CMD_DMA1_TX, curr_len, DMA1_tx_addr, dma_tag_tmp);
			DMA1_tx_addr += curr_len;
		}
		if( what_DMAS & USE_DMA1_TX_WITHOUT_TLAST){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd_without_EOF(CMD_DMA1_TX, len, DMA1_tx_addr, dma_tag_tmp);
			DMA1_tx_addr += curr_len;
		}
		if( what_DMAS & USE_DMA2_RX){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd(CMD_DMA2_RX, curr_len, DMA2_rx_addr, dma_tag_tmp);
			DMA2_rx_addr += curr_len;
		}
		remaining_to_move -= curr_len;
	}

	//verify dma response
	for (i = 0; remaining_to_ack > 0 &&  i < DMA_MAX_TRANSACTIONS; i++){
		if (remaining_to_ack > dma_transaction_size)
			curr_len = dma_transaction_size;
		else
			curr_len = remaining_to_ack;
		//wait for DMAs to finish
		count = 0;
		do {
			if(timeout != 0 && count >= timeout )
				longjmp(excp_handler, DMA_TIMEOUT_ERROR);
			count ++;
			invalid = 0;
			if( what_DMAS & USE_DMA0_RX){
				invalid += tngetd(STS_DMA0_RX);
			}
			if( what_DMAS & USE_DMA1_RX){
				invalid += tngetd(STS_DMA1_RX);
			}
			if( what_DMAS & (USE_DMA1_TX | USE_DMA1_TX_WITHOUT_TLAST)){
				invalid += tngetd(STS_DMA1_TX);
			}
			if( what_DMAS & USE_DMA2_RX){
				invalid += tngetd(STS_DMA2_RX);
			}
			
		} while (invalid);

		if( what_DMAS & USE_DMA0_RX){
			status = getd(STS_DMA0_RX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, curr_len, dma_tag, 0);
		}
		if( what_DMAS & USE_DMA1_RX){
			status = getd(STS_DMA1_RX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, curr_len, dma_tag, 0);
		}
		if( what_DMAS & USE_DMA1_TX){
			status = getd(STS_DMA1_TX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, curr_len, dma_tag, 1);
		}
		if( what_DMAS & USE_DMA1_TX_WITHOUT_TLAST){
			dma_tag_tmp = (dma_tag_tmp + 1) & 0xf;
			dma_cmd_without_EOF(CMD_DMA1_TX, len, DMA1_tx_addr, dma_tag_tmp);
			DMA1_tx_addr += curr_len;
		}
		if( what_DMAS & USE_DMA2_RX){
			status = getd(STS_DMA2_RX);
			dma_tag = (dma_tag + 1) & 0xf;
			check_DMA_status(status, curr_len, dma_tag, 0);
		}
		
		remaining_to_ack  -= curr_len;
		len			      -= curr_len;
		moved			  += curr_len;
	}
	return moved;
}

//before calling this method call cfg_switch
//segment a logical dma command in multiple dma commands.
//if one of the DMA is not used fill the corresponding address with NULL
static inline int dma_movement( 	unsigned int len,
					unsigned long long DMA0_rx_addr,
					unsigned long long DMA1_rx_addr,
					unsigned long long DMA1_tx_addr,
					unsigned long long DMA2_rx_addr,
					unsigned int what_DMAS) {
	unsigned int curr_len;
	//issue multiple DMA commands to move data around.
	//len keeps number of bytes still to transfer
	while ( len > 0){
		curr_len = dma_movement_DMA_MAX_TRANSACTIONS( len, DMA0_rx_addr, DMA1_rx_addr, DMA1_tx_addr, DMA2_rx_addr, what_DMAS);
		len			 -= curr_len;
		DMA0_rx_addr += curr_len;
		DMA1_rx_addr += curr_len;
		DMA1_tx_addr += curr_len;
		DMA2_rx_addr += curr_len;
	}
	
	return COLLECTIVE_OP_SUCCESS;
}


//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
int dma_loopback(	unsigned int len,
					unsigned long long src_addr,
					unsigned long long dst_addr) {
	
	//configure switch
	cfg_switch(DATAPATH_DMA_LOOPBACK, ARITH_NONE );
	//call function that will control the DMAs
	return dma_movement(len, 0, src_addr, dst_addr, 0, USE_DMA1_RX | USE_DMA1_TX);

}

//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
static inline int copy(	unsigned int len,
						unsigned long long src_addr,
						unsigned long long dst_addr) {
	//synonym
	return  dma_loopback(	  len, src_addr, dst_addr);
}

//performs a copy using DMA1. DMA1 rx reads while DMA1 tx overwrites
//note that this function has to be used when len < DMA_MAX_BTT
static inline int copy_single_dma_transaction(	unsigned int len,
						unsigned long long src_addr,
						unsigned long long dst_addr) {

	//configure switch
	cfg_switch(DATAPATH_DMA_LOOPBACK, ARITH_NONE );
	//start both channels of the DMA1
	return dma_movement_single(len, 0, src_addr, dst_addr, 0, USE_DMA1_RX | USE_DMA1_TX);

}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx back to dst buffer
int reduce_loopback(	unsigned int len,
						unsigned int func,
						unsigned long long op0_addr,
						unsigned long long op1_addr,
						unsigned long long dst_addr) {
	
	//configure switch
	cfg_switch(DATAPATH_DMA_REDUCTION, ARITH_INTERNAL);

	//start_arith(len, func, 0);
	////start dma transfer
	//unsigned int ret = dma_movement(len, op0_addr, op1_addr, dst_addr, 0, USE_DMA0_RX | USE_DMA1_RX | USE_DMA1_TX); //USE_DMA1_TX_WITHOUT_TLAST
	////wait for arith to finish
	//wait_arith();
	//
	//return ret;
	unsigned int moved;
	while(len > 0){
		start_arith(len, func, 0);
		//start dma transfer
		moved = dma_movement_DMA_MAX_TRANSACTIONS(len, op0_addr, op1_addr, dst_addr, 0, USE_DMA0_RX | USE_DMA1_RX | USE_DMA1_TX);
		//wait for arith to finish
		wait_arith();
		//
		len 	 -= moved;
		op0_addr += moved;
		op1_addr += moved;
		dst_addr += moved;
	}
	return COLLECTIVE_OP_SUCCESS;
}

//performs an accumulate using DMA1 and DMA0. DMA0 rx reads op1 DMA1 rx reads op2 while DMA1 tx overwrites op2 buffer
static inline int accumulate (	
						unsigned int len,
						unsigned int func,
						unsigned long long op1_addr,
						unsigned long long op2_addr,
						unsigned long long dst_addr) {
	return reduce_loopback(	 len, func, op1_addr, op2_addr, dst_addr);
}

//transmits a buffer to a rank of the world communicator
int transmit_offchip_world(
	communicator *world,
	unsigned int dst_rank,
	unsigned int len,
	unsigned long long src_addr,
	unsigned int dst_tag
){
	unsigned int bytes_moved, messages;

	cfg_switch(use_tcp ? DATAPATH_OFFCHIP_TX_TCP : DATAPATH_OFFCHIP_TX_UDP , ARITH_NONE);
	//divide and send msg in multiple transactions each with a different id in the sequence 
	//and ack them separately
	while(len > 0){
		messages 	= start_packetizer_message(world, dst_rank, len		, dst_tag);
		bytes_moved = dma_movement_DMA_MAX_TRANSACTIONS(len, 0, src_addr, 0, 0, USE_DMA1_RX);
		ack_packetizer(world, dst_rank, messages);

		len 	 -= bytes_moved;
		src_addr += bytes_moved;
	}
	return COLLECTIVE_OP_SUCCESS;
}

//transmits a buffer to a rank of the comm-esim communicator 
static inline int transmit_offchip(	
						unsigned int comm,
						unsigned int dst_rank,
						unsigned int len,
						unsigned long long src_addr,
						unsigned int dst_tag
					) {
	//find communicator
	communicator world = find_comm(comm);
	return transmit_offchip_world(&world, dst_rank, len, src_addr, dst_tag);
}

//performs an accumulate using DMA0 and DMA1 and internal arith unit. Then it forwards the result 
//through tx_subsystem to dst_rank
int reduce_offchip_world(
	communicator *world,
	unsigned int dst_rank,
	unsigned int len, 
	unsigned int func,
	unsigned long long op0_addr,
	unsigned long long op1_addr,
	unsigned int dst_tag
){
	unsigned int moved;
	//configure switch
	cfg_switch(use_tcp ? DATAPATH_OFFCHIP_TCP_REDUCTION : DATAPATH_OFFCHIP_UDP_REDUCTION, ARITH_INTERNAL);
	/*
	start_arith(len, func, 0);
	ret = dma_movement(len, op0_addr, op1_addr, 0, 0, USE_DMA0_RX | USE_DMA1_RX );	
	wait_arith();
	*/
	while(len > 0){
		start_arith(len, func, 0);
		//start dma transfer
		moved = dma_movement_DMA_MAX_TRANSACTIONS(len, op0_addr, op1_addr, 0, 0, USE_DMA0_RX | USE_DMA1_RX );
		//wait for arith to finish
		wait_arith();
		//
		len 	 -= moved;
		op0_addr += moved;
		op1_addr += moved;
	}
	return COLLECTIVE_OP_SUCCESS;

}

//performs an accumulate using DMA0 and DMA1 and then forwards the result 
//through tx_subsystem to dst_rank
static inline int reduce_offchip(	
					unsigned int comm,
					unsigned int dst_rank,
					unsigned int len, 
					unsigned int func,
					unsigned long long op0_addr,
					unsigned long long op1_addr,
					unsigned int dst_tag) {
	
	//find communicator
	communicator world = find_comm(comm);
	//synonym function
	return reduce_offchip_world(&world, dst_rank, len, func, op0_addr, op1_addr, dst_tag);
}

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
		//TODO:speedup new_dma tag using dma_tag variable
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

//iterates over rx buffers until match is found or a timeout expires
//matches len, src and tag if tag is not ANY
//returns the index of the spare_buffer
//timeout is jumps to exception handler
int wait_receive_world_i(	
						communicator *world,	
						unsigned int src_rank,
						unsigned int len,
						unsigned int src_tag
					){
	unsigned int seq_num, i; //src_port TODO: use this variable to choose depending on tcp/udp session id or port
	seq_num 	= world->ranks[src_rank].inbound_seq + 1;
	//TODO: use a list to store recent message received to avoid scanning in the entire spare_buffer_struct.
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

//waits for a messages to come and move their contents in a buffer
int receive_offchip_world(
						communicator *world,	
						unsigned int src_rank,	
						unsigned int len,
						unsigned long long dst_addr,
						unsigned int src_tag
						){
	unsigned long long rx_buff_addr;
	unsigned int curr_len;
	unsigned int ret=COLLECTIVE_OP_SUCCESS;
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);

	//default fullsize transfer
	curr_len = dma_transaction_size;
	for (; len > 0 && ret == COLLECTIVE_OP_SUCCESS ;  len -= curr_len, dst_addr+=curr_len){
		if(len < dma_transaction_size){
			curr_len = len;
		}	

		int buf_idx = wait_receive_world_i(world, src_rank, curr_len, src_tag);
		if  (buf_idx < 0 )
			return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		rx_buff_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
		ret = copy_single_dma_transaction(curr_len, rx_buff_addr, dst_addr);
		//set spare buffer as reserved (i.e. valid and not in DMA0 queue)
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();
	}

	return ret;
}

//1) receives from a rank 
//2) sums with a a buffer 
//3) the result is saved in (possibly another) local buffer
int receive_and_accumulate(
		communicator *world,
		unsigned int src_rank,
		unsigned int len, 
		unsigned int func,
		unsigned long long op0_addr,
		unsigned long long dst_addr,
		unsigned int dst_tag)
	{
	unsigned long long buf_addr;
	unsigned int curr_len, buf_idx;
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//configure switch
	cfg_switch(DATAPATH_DMA_REDUCTION, ARITH_INTERNAL);
	//default fullsize transfer
	curr_len = dma_transaction_size;
	for (; len > 0 ; len -= curr_len, op0_addr += curr_len,	dst_addr += curr_len ){
		if(len < dma_transaction_size ) {
			//last transaction
			curr_len = len;
		}	
		//start arithmetic unit 
		start_arith(curr_len, func, 0);
		//1. receive part of data from previous in rank and retrieve spare buffer index
		buf_idx = wait_receive_world_i(world, src_rank, curr_len, TAG_ANY);
		if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		buf_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
		//start dma transfer
		dma_movement_single(curr_len, op0_addr, buf_addr, dst_addr, 0, USE_DMA0_RX | USE_DMA1_RX | USE_DMA1_TX);
		
		//3. release spare buffer
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();
 
		wait_arith();
	}
	return COLLECTIVE_OP_SUCCESS;
}

//1) receives from a rank 
//2) sums with a a buffer
//3) result is sent to another rank
int receive_and_reduce_offchip(
		communicator *world,
		unsigned int src_rank,
		unsigned int dst_rank,
		unsigned int len, 
		unsigned int func,
		unsigned long long op0_addr,
		unsigned int dst_tag)
	{
	unsigned long long buf_addr;
	unsigned int curr_len, buf_idx, messages;
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//configure switch
	cfg_switch( use_tcp ? DATAPATH_OFFCHIP_TCP_REDUCTION : DATAPATH_OFFCHIP_UDP_REDUCTION, ARITH_INTERNAL);

	//default fullsize transfer
	curr_len = dma_transaction_size;
	for (; len > 0 ; len -= curr_len, op0_addr += curr_len){
		if(len < dma_transaction_size) {
			//last transaction
			curr_len = len;
		}	
		//start arithmetic unit 
		start_arith(curr_len, func, 0);
		//1. receive part of data from previous in rank and retrieve spare buffer index
		buf_idx = wait_receive_world_i(world, src_rank, curr_len, TAG_ANY);
		if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
		buf_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
		//start packetizer
		messages = start_packetizer_message(world, dst_rank, curr_len, TAG_ANY);
		//start dma transfer
		dma_movement_single(curr_len, op0_addr, buf_addr, 0, 0, USE_DMA0_RX | USE_DMA1_RX );
		//3. release spare buffer
		microblaze_disable_interrupts();
		rx_buf_list[buf_idx].status = STATUS_IDLE;
		microblaze_enable_interrupts();
		wait_arith();
		ack_packetizer(world, dst_rank, messages);
	}
	return COLLECTIVE_OP_SUCCESS;
}

//receives from a rank and forwards to another rank 
int relay_to_other_rank(
		communicator *world,
		unsigned int src_rank,
		unsigned int dst_rank,
		unsigned int len, 
		unsigned int dst_tag){
		
		unsigned int curr_len, buf_idx, messages;
		unsigned long long buf_addr;
		rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
		//configure switch
		cfg_switch( use_tcp ? DATAPATH_OFFCHIP_TX_TCP: DATAPATH_OFFCHIP_TX_UDP, ARITH_NONE);

		for (; len > 0 ; len -= curr_len ){
			if (len < dma_transaction_size)
				curr_len = len;
			else	
				curr_len = dma_transaction_size;

			//1. receive part of data from previous in rank and retrieve spare buffer index
			buf_idx = wait_receive_world_i(world, src_rank, curr_len, TAG_ANY);
			if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
			//2. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
			//configure switch
			buf_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;
			messages = start_packetizer_message(world, dst_rank, curr_len, TAG_ANY);
			//start dma transfer
			dma_movement_single(curr_len, 0, buf_addr, 0, 0, USE_DMA1_RX );
			//3. release spare buffer
			microblaze_disable_interrupts();
			rx_buf_list[buf_idx].status = STATUS_IDLE;
			microblaze_enable_interrupts();
			ack_packetizer(world, dst_rank, messages);
		}
		return COLLECTIVE_OP_SUCCESS;

}

static inline int send(		
				unsigned int comm,
				unsigned int len,
				unsigned int tag,
				unsigned int dst_rank,
				unsigned long long buf_addr){
	//send synonym
	return transmit_offchip(comm, dst_rank, len, buf_addr, tag);
}

static inline int recv(		
				unsigned int comm,
				unsigned int len,
				unsigned int tag,
				unsigned int src_rank,
				unsigned long long buf_addr){
	//find communicator
	communicator world = find_comm(comm);
	return receive_offchip_world(&world, src_rank, len, buf_addr,  tag);
}

//naive bcast: root sends to each rank
int broadcast(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned long long buf_addr){
	int i;
	int ret = COLLECTIVE_OP_SUCCESS;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		//send to all other members of the communicator
		for(i=0; i<world.size && ret == COLLECTIVE_OP_SUCCESS ; i++){
			if(i != world.local_rank){
				ret = transmit_offchip_world(&world, i		, len, buf_addr, TAG_ANY);
			}
		}
	}else{
		ret = 	receive_offchip_world(	&world, src_rank,	  len, buf_addr, TAG_ANY);
	}
	return ret;
}

//root read multiple times the same segment and send it to each rank before moving to the next part of the buffer to be transmitted. 
int broadcast_round_robin(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned long long buf_addr){
	unsigned int curr_len, i;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		
		if (use_tcp){
			cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_NONE);
		}else{
			cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_NONE);
		}
		//send to all members of the communicator 1 segment of the buffer at the time
		curr_len = dma_transaction_size;
		while (len > 0)
		{
			if(len < dma_transaction_size)
				curr_len = len;
			for(i=0; i<world.size; i++){
				if(i == world.local_rank) continue;
				//start packetizer messages and issue a single dma command for eachrank
				start_packetizer_message(&world, i, curr_len, TAG_ANY);
				dma_movement_single(curr_len, 0, buf_addr, 0, 0, USE_DMA1_RX);
			}
			//ack packetizer 
			for(i=0; i<world.size; i++){
				if(i== world.local_rank) continue;
				
				ack_packetizer(&world, i, 1);
			} 
			//move on in the buffer
			buf_addr += curr_len;
			len 	 -= curr_len;
		}
		return COLLECTIVE_OP_SUCCESS;
	}else{
		return receive_offchip_world(	&world, src_rank, len, buf_addr, TAG_ANY);
	}
	
}

//naive: root sends to each rank the buffer.
int scatter(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned long long src_buf_addr,
				unsigned long long dst_buf_addr){
	int i;
	int ret = COLLECTIVE_OP_SUCCESS;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		//for root rank configure switch
		for(i = 0; i < world.size && ret == COLLECTIVE_OP_SUCCESS; i++){
			if (i == world.local_rank){
				//for root copy
				ret = copy(	len, src_buf_addr, dst_buf_addr);
			}else{
				ret = transmit_offchip_world(&world, i, len, src_buf_addr, TAG_ANY);
			}
			src_buf_addr += len;
		}
	}else{
		return receive_offchip_world(			&world, src_rank, len, dst_buf_addr,  TAG_ANY);
	}

	return ret;
}

//scatter segment at a time. root sends each rank a segment in a round robin fashion 
int scatter_rr(	
				unsigned int comm,
				unsigned int len,
				unsigned int src_rank,
				unsigned long long src_buf_addr,
				unsigned long long dst_buf_addr){
	unsigned int curr_len, i, offset_next_rank_buffer;
	unsigned long long tmp_src_buf_base_addr, tmp_src_buf_offset_addr;
	offset_next_rank_buffer = len;
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(src_rank == world.local_rank){
		
		if (use_tcp){
			cfg_switch(DATAPATH_OFFCHIP_TX_TCP, ARITH_NONE);
		}else{
			cfg_switch(DATAPATH_OFFCHIP_TX_UDP, ARITH_NONE);
		}
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
					start_packetizer_message(&world, i, curr_len, TAG_ANY);
					dma_movement_single(curr_len, 0, tmp_src_buf_offset_addr, 0, 0, USE_DMA1_RX);
				}
				tmp_src_buf_offset_addr += offset_next_rank_buffer;
			}
			//ack packetizer 
			for(i=0; i<world.size; i++){
				if(i== world.local_rank) continue;
				
				ack_packetizer(&world, i, 1);
			} 
			//move on in the buffer
			tmp_src_buf_base_addr += curr_len;
			len 	 	 		  -= curr_len;
		}

		//for root copy
		src_buf_addr += world.local_rank*offset_next_rank_buffer;
		copy(	offset_next_rank_buffer, src_buf_addr, dst_buf_addr);

		return COLLECTIVE_OP_SUCCESS;
	}else{
		return receive_offchip_world(			&world, src_rank, len, dst_buf_addr,  TAG_ANY);
	}

}
//naive gather: every rank send to root. Root place them in the correct part of dst buffer.
int gather(		
				unsigned int comm,
				unsigned int len,
				unsigned int root_rank,
				unsigned long long src_buf_addr,
				unsigned long long dst_buf_addr){
	int i;
	int ret = COLLECTIVE_OP_SUCCESS;
	return COLLECTIVE_NOT_IMPLEMENTED; //at this moment is not safe to run non ring based collectives. they are not handled at depacketizer level.
	//find communicator
	communicator world = find_comm(comm);
	//determine if we're sending or receiving
	if(root_rank == world.local_rank){
		//receive from all members of the communicator
		for(i=0; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++){
			//TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
			//currently this will go sequentially through the segments, which is slower and also requires more spare RX buffers
			if(i==world.local_rank)
			{ // root copies
				ret	= copy(				len, src_buf_addr, dst_buf_addr);
			}else{
				ret = receive_offchip_world(	&world, i		 , len, dst_buf_addr, TAG_ANY);
			}
			dst_buf_addr += len;
		}
	}else{
		ret = 		transmit_offchip_world(		&world, root_rank, len, src_buf_addr, TAG_ANY);
	}
	return ret;
}

//naive gather: non root relay data to the root. root copy segments in dst buffer as they come.
int gather_ring(
				unsigned int comm,
				unsigned int len,
				unsigned int root_rank,
				unsigned long long src_buf_addr,
				unsigned long long dst_buf_addr){
	unsigned long long tmp_buf_addr;
	unsigned int i,curr_pos, next_in_ring, prev_in_ring, number_of_shift;
	int ret = COLLECTIVE_OP_SUCCESS;
	//find communicator
	communicator world = find_comm(comm);
	next_in_ring = (world.local_rank+1			   ) % world.size	;
	prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	
	if(root_rank == world.local_rank){ //root ranks mainly receives

		//receive from all members of the communicator
		for(i=0, curr_pos = prev_in_ring; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
			//TODO: optimize receives; what we want is to move any data which arrives into the correct segment of the buffer
			//currently this will go sequentially through the segments, which is slow and also requires more spare RX buffers
			tmp_buf_addr  = dst_buf_addr + len * curr_pos;
			if(curr_pos==world.local_rank)
			{ // root copies
				ret	= copy(	     len, src_buf_addr, tmp_buf_addr);
			}else{
				ret = receive_offchip_world(&world, prev_in_ring, len, tmp_buf_addr, TAG_ANY);
			}
		}
	}else{
		//non root ranks sends their data + relay others data to the next rank in sequence
		// as a daisy chain
		number_of_shift = ((world.size+world.local_rank-root_rank)%world.size) - 1 ; //distance to the root
		ret += transmit_offchip_world(		&world, next_in_ring, len, src_buf_addr, TAG_ANY);
		for (int i = 0; i < number_of_shift; i++)
		{	
			//relay the others 
			relay_to_other_rank(&world, prev_in_ring, next_in_ring, len, TAG_ANY);
		}	
	}
	return ret;
}

//naive: gather + bcast
static inline int allgather(	
				unsigned int comm,
				unsigned int len,
				unsigned int internal_root,
				unsigned long long src_addr,
				unsigned long long dst_addr){	
	int ret = 0;
	//find communicator
	communicator world = find_comm(comm);
	ret = gather_ring(	comm, len			, internal_root, src_addr, dst_addr);
	if (ret != COLLECTIVE_OP_SUCCESS ) return ret;
	return broadcast(	comm, len*world.size, internal_root, dst_addr);
}

//naive fused: 1) receive a segment 2) move in the dest buffer 3) relay to next rank 
int allgather_ring(
				unsigned int comm,
				unsigned int len,
				unsigned long long src_buf_addr,
				unsigned long long dst_buf_addr){
	unsigned long long tmp_buf_addr;
	unsigned int i,curr_pos, next_in_ring, prev_in_ring;
	int ret = COLLECTIVE_OP_SUCCESS;
	//find communicator
	communicator world = find_comm(comm);
	next_in_ring = (world.local_rank+1			   ) % world.size	;
	prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	//load buf_addr

	//send our data to next in ring
	ret += transmit_offchip_world(				&world, next_in_ring,len, src_buf_addr, TAG_ANY);
	//receive from all members of the communicator
	for(i=0, curr_pos = world.local_rank; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++, curr_pos = (curr_pos + world.size - 1 ) % world.size){
		tmp_buf_addr  = dst_buf_addr + len * curr_pos;

		if(curr_pos==world.local_rank){
			ret	= copy(len, src_buf_addr, tmp_buf_addr);
		}else{
			ret = receive_offchip_world(	  	&world, prev_in_ring, len, tmp_buf_addr, TAG_ANY);
			//TODO: use the same stream to move data both to dst_buf_addr and tx_subsystem 
			//todo: use DMA1 RX to forward to next and DMA0/2 RX and DMA1 TX to copy ~ at the same time! 
			if(i+1 < world.size){ //if not the last data needed relay to the next in the sequence
				ret = transmit_offchip_world(	&world, next_in_ring, len, tmp_buf_addr, TAG_ANY);
			}
		}		
	}

	return ret;
}

//naive. every rank forwards its buffer they are received and sum by the root
int reduce(		
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int root_rank,
				unsigned long long src_addr,
				unsigned long long dst_addr
			){
	return COLLECTIVE_NOT_IMPLEMENTED; //at this moment is not safe to run non ring based collectives. they are not handled at depacketizer level.
	int ret = COLLECTIVE_OP_SUCCESS;
	int buf_idx;
	long long buf_addr;
	//find communicator
	communicator world = find_comm(comm);
	//get rx_buff_location //TODO: write a function for that or move it into a static variable
	unsigned int nbufs 	   = 	 Xil_In32(RX_BUFFER_COUNT_OFFSET);
	rx_buffer *rx_buf_list = (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	//determine if we're sending or receiving
	if(root_rank == world.local_rank){
		//0. copy src_buffer of master in dst_buffer
		// from now on dst_buffer will represent the accumulator
		ret += copy(len, src_addr, dst_addr);
		//receive and accumulate from all members of the communicator
		for(int i=0; i<world.size && ret == COLLECTIVE_OP_SUCCESS; i++){
			//0. skip if we are root rank
			if(i == root_rank)
				continue;
			//1. receive part of data and retrieve spare buffer index
			buf_idx = wait_receive_world_i(&world, i, len, TAG_ANY);
			if  (buf_idx < 0 || buf_idx >= nbufs )
				return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
			//2. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
			//configure switch
			buf_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl;

			ret = accumulate(len, function, buf_addr, dst_addr,  dst_addr);
			//3. release buffer
			microblaze_disable_interrupts();
			rx_buf_list[buf_idx].status = STATUS_IDLE;
			microblaze_enable_interrupts();
			//move to next rank
		}
	}else{
		ret += transmit_offchip_world(&world, root_rank, len, src_addr,  TAG_ANY);
	}
	return ret;
}

//every rank receives a buffer it sums its own buffer and forwards to next rank in the ring
int reduce_ring_streaming(
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int root_rank,
				unsigned long long src_addr,
				unsigned long long dst_addr){
	int ret;
	//find communicator
	communicator world 		  = find_comm(comm);
	unsigned int next_in_ring = (world.local_rank+1			   ) % world.size	;
	unsigned int prev_in_ring = (world.local_rank+world.size-1 ) % world.size	;
	//determine if we're sending or receiving
	if( prev_in_ring == root_rank){ 
		//non root ranks immediately after the root sends
		ret = transmit_offchip_world(&world, next_in_ring, len, src_addr, TAG_ANY);
	}else if (world.local_rank != root_rank){
		//non root ranks sends their data + data received from previous rank to the next rank in sequence as a daisy chain
		ret = receive_and_reduce_offchip(&world, prev_in_ring, next_in_ring, len, function, src_addr,  TAG_ANY);
	}else{	
		//root only receive from previous node tin the ring, add its local buffer and save in destination buffer 
		ret = receive_and_accumulate(&world, prev_in_ring, len, function, src_addr, dst_addr, TAG_ANY);
	}
	return ret;
}

//naive non fused allreduce: reduce+broadcast
int allreduce(	
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned int internal_root,
				unsigned long long src_addr,
				unsigned long long dst_addr){

	int ret = 0;
	//let 0 be the root
	ret = reduce_ring_streaming(comm, len, function, internal_root, src_addr, dst_addr);
	if (ret != COLLECTIVE_OP_SUCCESS ) return ret;
	return broadcast(			  comm, len,		   internal_root, dst_addr);
}

//naive: relay buffers along the ring and each rank accumulates them.
int allreduce_fused_ring(	
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned long long src_addr,
				unsigned long long dst_addr){

	int ret;
	//find communicator
	communicator world 		  	= find_comm(comm);
	unsigned int next_in_ring 	= (world.local_rank+1			   )%world.size	;
	unsigned int prev_in_ring 	= (world.local_rank+world.size-1 )%world.size	;
	//get rx_buff_location
	rx_buffer *rx_buf_list 	  	= (rx_buffer*)(RX_BUFFER_COUNT_OFFSET+4);
	unsigned int buf_idx, tmp_len, curr_len;
	unsigned long long buf_addr, tmp_addr; 
	//every member of the communicator 
	// 0.  sends its data to the next rank
	// 1. put their own data in their buffer
	// then receive others data:
	// 2.  receives data from previous rank
	// 3.  relay data to the next in sequence
	// 4.  accumulates data in the destination buffer
	// 0.send our data
	ret = copy(len, src_addr, dst_addr);
	if (ret != COLLECTIVE_OP_SUCCESS) return ret;

	//1. copy src_buffer of master in dst_buffer
	// from now on dst_buffer will represent the accumulator
	ret = transmit_offchip_world(&world, next_in_ring,			len, src_addr, TAG_ANY);
	for(int i=0; i < world.size-1 && ret == COLLECTIVE_OP_SUCCESS; i++){
		tmp_addr = dst_addr;
		curr_len = dma_transaction_size;
		for(tmp_len = len; tmp_len > 0 && ret == COLLECTIVE_OP_SUCCESS; tmp_len-=curr_len, tmp_addr+=curr_len){
			if(tmp_len < dma_transaction_size)
				curr_len = tmp_len;
			//2. receive part of data from previous rank and retrieve spare buffer index
			buf_idx = wait_receive_world_i(&world, prev_in_ring, curr_len, TAG_ANY);
			if  (buf_idx < 0 ) return RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID;
			buf_addr = ((long long) rx_buf_list[buf_idx].addrh << 32) | rx_buf_list[buf_idx].addrl; 
			//3. relay others data only if it's not the last iteration.  
			if ( i+1<world.size-1){
				ret = transmit_offchip_world(&world, next_in_ring, curr_len, buf_addr, TAG_ANY);
			}
			//4. accumulate in buffer: user_buffer = user_buffer + spare_buffer 
			if (ret != COLLECTIVE_OP_SUCCESS) return ret;
			ret = accumulate(   curr_len, function, buf_addr, tmp_addr, tmp_addr);
			//housekeeping: release spare buffer
			microblaze_disable_interrupts();
			rx_buf_list[buf_idx].status = STATUS_IDLE;
			microblaze_enable_interrupts();
			//TODO: use reduce_offchip when using tree based collectives
		}
	}
	return ret;
}

static inline int arith_number_of_bytes_per(unsigned int function){
	switch (function)
	{
		case ARITH_fp:
			return 4;
		case ARITH_dp:
			return 8;
		case ARITH_i32:
			return 4;
		case ARITH_i64:
			return 8;
	}
	return -1;
}

//2 stage allreduce: distribute sums across the ranks. smaller bandwidth between ranks and use multiple arith at the same time.
int all_reduce_share(
				unsigned int comm,
				unsigned int len,
				unsigned int function,
				unsigned long long src_addr,
				unsigned long long dst_addr){
	unsigned int ret,i;
	uint64_t curr_recv_addr, curr_send_addr;
	//find communicator
	communicator world 		  	 = find_comm(comm);
	unsigned int next_in_ring 	 = (world.local_rank+1			  )%world.size	;
	unsigned int prev_in_ring 	 = (world.local_rank+world.size-1 )%world.size	;
	//divide into chunks a
	unsigned int len_div_size 	 = len/world.size; 
	unsigned int tail 		     = len_div_size % min_operand_width;
	unsigned int min_operand_width = arith_number_of_bytes_per(function);
	len_div_size += - tail + (tail == 0 ? 0 : min_operand_width); //TODO: move

	unsigned int curr_send_chunk = world.local_rank;
	unsigned int curr_recv_chunk = (curr_send_chunk + world.size - 1) % world.size;

	//A) Share reduce stage
	//1. each rank send its own chunk to the next rank in the ring
	curr_send_addr = src_addr + len_div_size * curr_send_chunk;
	ret = transmit_offchip_world(&world, next_in_ring,	len_div_size, curr_send_addr, TAG_ANY);
	if(ret != COLLECTIVE_OP_SUCCESS) return ret;
	//2. for n-1 times sum the chunk that is coming from previous rank with your chunk at the same location. then forward the result to
	// the next rank in the ring
	for (i = 0; i < world.size -2; ++i, curr_recv_chunk=(curr_recv_chunk + world.size - 1) % world.size)
	{
		curr_recv_addr = src_addr + len_div_size * curr_recv_chunk;
		//2. receive part of data from previous in rank and accumulate to the part you have at the same address
		//ad forwar dto the next rank in the ring
		ret = receive_and_reduce_offchip(&world, prev_in_ring, next_in_ring, len_div_size, function, curr_recv_addr, TAG_ANY); 
		if(ret != COLLECTIVE_OP_SUCCESS) return ret;
	}
	//at last iteration (n-1) you can sum and save the result at the same location (which is the right place to be)
	if (i > 0 ){
		curr_recv_addr  = src_addr + len_div_size * curr_recv_chunk; 
		curr_send_addr  = dst_addr + len_div_size * curr_recv_chunk;
		ret = receive_and_accumulate(&world, prev_in_ring, len_div_size, function, curr_recv_addr, curr_send_addr, TAG_ANY); 
	}

	//B) Share result stage at this point each of the ranks has at curr_recv_chunk the final result.
	//they have to share it with the others so that all have the entire buffer
	for (i = 0; i < world.size -1; ++i)
	{
		curr_send_chunk =  curr_recv_chunk;
		curr_recv_chunk = (curr_recv_chunk + world.size - 1) % world.size;
		curr_send_addr  = dst_addr + len_div_size * curr_send_chunk;
		curr_recv_addr  = dst_addr + len_div_size * curr_recv_chunk;
		//TODO: 5.6. can be done together since we have 3 dmas (1rx read from spare, 1tx write in the final dest, 2rx can read and transmit to next in the ring)
		//5. send the local reduced results to next rank
		ret = transmit_offchip_world(&world, next_in_ring,	len_div_size, curr_send_addr, TAG_ANY);
		if(ret != COLLECTIVE_OP_SUCCESS) return ret;
		//6. receive the reduced results from previous rank and put into correct dst buffer position
		ret = receive_offchip_world(&world, prev_in_ring,	len_div_size, curr_recv_addr, TAG_ANY);
		if(ret != COLLECTIVE_OP_SUCCESS) return ret;
	}

	return COLLECTIVE_OP_SUCCESS;
}



int ext_kernel_stream(
						unsigned int len,
						unsigned long long src_addr,
						unsigned long long dst_addr
){
	//configure switch
	cfg_switch(DATAPATH_DMA_EXT_LOOPBACK, ARITH_NONE);
	return dma_movement(len, 0, src_addr, dst_addr, 0, USE_DMA1_RX | USE_DMA1_TX);
}

int reduce_ext (	unsigned int len,
					unsigned long long op1_addr,
					unsigned long long op2_addr,
					unsigned long long dst_addr
				) {
	
	//configure switch
	cfg_switch(DATAPATH_DMA_REDUCTION, ARITH_EXTERNAL);
	return dma_movement(len, op1_addr, op2_addr, dst_addr, 0, USE_DMA0_RX | USE_DMA1_RX | USE_DMA1_TX);
	
}


int main() {
	unsigned int retval;
	unsigned int scenario, len, comm, root_src_dst, function, msg_tag;
	unsigned int buf0_type, buf1_type, buf2_type;
	unsigned int buf0_addrl, buf0_addrh, buf1_addrl, buf1_addrh, buf2_addrl, buf2_addrh;
	unsigned long long buf0_addr, buf1_addr, buf2_addr;

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
		buf0_addr =  ((long long) buf0_addrh << 32) | buf0_addrl;
		buf1_addr =  ((long long) buf1_addrh << 32) | buf1_addrl;
		buf2_addr =  ((long long) buf2_addrh << 32) | buf2_addrl;
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
					case SET_DMA_TRANSACTION_SIZE:
						dma_transaction_size = len;
						start_timer1();
						Xil_Out32(TIME_TO_ACCESS_EXCH_MEM, read_timer1());
						break;
					default:
						break;
				}
				
				break;
			case XCCL_SEND:
				retval = send(					comm, len, msg_tag,  root_src_dst, 	buf0_addr);
				break;
			case XCCL_RECV:
				retval = recv(					comm, len, msg_tag,  root_src_dst, 	buf0_addr);
				break;
			case XCCL_BCAST:
				retval = broadcast(				comm, len, 			 root_src_dst, 	buf0_addr);
				break;
			case XCCL_BCAST_RR:
				retval = broadcast_round_robin( comm, len, 			 root_src_dst, 	buf0_addr);
				break;
			case XCCL_SCATTER:	
				retval = scatter(				comm, len, 			 root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_SCATTER_RR:	
				retval = scatter_rr(			comm, len, 			 root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_GATHER:
				retval = gather(				comm, len, 			 root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_REDUCE:
				retval = reduce(				comm, len, function, root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_ALLGATHER:
				retval = allgather(				comm, len, 		     root_src_dst,	buf0_addr, buf1_addr);
				break;
			case XCCL_ALLREDUCE:
				retval = allreduce(				comm, len, function, root_src_dst,  buf0_addr, buf1_addr );
				break;
			case XCCL_ACC:
				retval = accumulate( 	  	  	  	  len, function, 				buf0_addr, buf1_addr, buf1_addr);
				break;
			case XCCL_COPY:
				retval = copy(	  		  	  	  	  len, 							buf0_addr, buf1_addr);
				break;
			case XCCL_REDUCE_RING:
				retval = reduce_ring_streaming( comm, len, function, root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_ALLREDUCE_FUSED_RING:
				retval = allreduce_fused_ring(	comm, len, function, 				buf0_addr, buf1_addr);
				break;
			case XCCL_ALLREDUCE_SHARE_RING:
				retval = all_reduce_share	(	comm, len, function, 				buf0_addr, buf1_addr);
				break;
			case XCCL_GATHER_RING:
				retval = gather_ring(			comm, len, 			 root_src_dst, 	buf0_addr, buf1_addr);
				break;
			case XCCL_ALLGATHER_RING:
				retval = allgather_ring(		comm, len, 							buf0_addr, buf1_addr);
				break;
			case XCCL_EXT_STREAM_KRNL:
				retval = ext_kernel_stream( 		  len, 			 				buf0_addr, buf1_addr);
				break;
			case XCCL_EXT_REDUCE:
				retval = reduce_ext(				  len, 			 				buf0_addr, buf1_addr, buf2_addr);
				break;
			default:
				retval = 0;
				break;
		}

		finalize_call(retval);
	}
	return 0;
}