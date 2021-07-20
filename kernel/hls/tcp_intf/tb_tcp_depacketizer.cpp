
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

using namespace hls;
using namespace std;

#define DATA_WIDTH 512

void tcp_depacketizer(	stream<ap_axiu<DATA_WIDTH,0,0,0> > & in,
			stream<ap_axiu<DATA_WIDTH,0,0,0> > & out,
			stream<ap_uint<32> > & sts);

int main(){

	stream<ap_axiu<DATA_WIDTH,0,0,0> > in;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > out;
	stream<ap_axiu<DATA_WIDTH,0,0,0> > golden;
	stream<ap_uint<32> > sts;
	
	ap_axiu<DATA_WIDTH,0,0,0> inword;
	ap_axiu<DATA_WIDTH,0,0,0> outword;
	ap_axiu<DATA_WIDTH,0,0,0> goldenword;
	
	int dest 	= 3;
	int len;
	int tag 	= 5;
	int src 	= 6;
	int seq		= 42

	//1024B+64B transfer
	len = 2048;

	for(int i=0; i<(len/64+1); i++){
		if(i==0){
			inword.data(31, 0) = len;
			inword.data(63,32) = tag;
			inword.data(95,64) = src;
			inword.data(127,96)= seq;
		} else {
			inword.data = i;
		}
		inword.last = (i%3 == 0) || (i==len/64);
		in.write(inword);
		if(i > 0)
			golden.write(inword);
	}
	
	tcp_depacketizer(in, out, sts);
	
	//parse header
	if(sts.read() != len) return 1;
	if(sts.read() != tag) return 1;
	if(sts.read() != src) return 1;
	if(sts.read() != seq) return 1;

	//parse data
	for(int i=0; i<len/64; i++){
		outword = out.read();
		goldenword = golden.read();
		if(outword.data != goldenword.data) return 1;
		int last = outword.last;
		if(i==len/64-1){
			if(last == 0) return 1;
		} else {
			if(last == 1) return 1;
		}
	}
	
	
	return 0;
}
