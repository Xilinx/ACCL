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

#include "xsi_dut.h"
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace std;

namespace {
    Log *logger;
}

XSI_DUT::XSI_DUT(const string& design_libname, const string& simkernel_libname,
                    const string& reset_name, bool reset_active_low,
                    const string& clock_name, float clock_period_ns, const string& wdbName, Log &log, bool trace) :
    xsi(design_libname, simkernel_libname)
{
    logger = &log;
    *logger << log_level::verbose << "Constructing DUT" << std::endl;
    //xsi = new Xsi::Loader(design_libname, simkernel_libname);
    *logger << log_level::verbose << "Created loader" << std::endl;
    //initialize XSI
    s_xsi_setup_info info;
    memset(&info, 0, sizeof(info));
    info.logFileName = NULL;
    info.wdbFileName = const_cast<char*>(wdbName.c_str());
    
    // Work-around for an added field (ximDir) in Vivado >= 2023.2, which if left NULL, will cause C++ string exception
    // However, for older versions of Vivado, this field is not present, so we need to check the size of the struct
    if (sizeof(info) == 24) {
        info.xsimDir =  const_cast<char*>("xsim.dir");
    }
    xsi.open(&info);
    *logger << log_level::verbose << "XSI opened" << std::endl;
    if(trace){
        *logger << log_level::verbose << "XSI waveform enabled" << std::endl;
        xsi.trace_all();
    }
    //get ports
    for(int i=0; i<xsi.get_num_ports(); i++){
        string port_name(xsi.get_port_name(i));
        port_parameters p = {i, xsi.get_port_bits(i), xsi.port_is_input(i)};
        port_map[port_name] = p;
    }
    //check clock and reset
    if(port_map.find(reset_name) == port_map.end()) throw invalid_argument("Reset not found in ports list");
    if(port_map[reset_name].port_bits != 1) throw invalid_argument("Reset is not a scalar");
    if(!port_map[reset_name].is_input) throw invalid_argument("Reset is not an input port");
    rst = reset_name;
    rst_active_low = reset_active_low;
    if(port_map.find(clock_name) == port_map.end()) throw invalid_argument("Clock not found in ports list");
    if(port_map[clock_name].port_bits != 1) throw invalid_argument("Clock is not a scalar");
    if(!port_map[clock_name].is_input) throw invalid_argument("Clock is not an input port");
    clk = clock_name;
    clk_half_period = (unsigned int)(clock_period_ns*pow(10,-9)/xsi.get_time_precision()/2);
    if(clk_half_period == 0) throw invalid_argument("Calculated half period is zero");
    //report results
    *logger << log_level::verbose << "Identified " << num_ports() << " top-level ports:" << endl;
    list_ports();
    *logger << log_level::verbose << "Using " << rst << " as " << (rst_active_low ? "active-low" : "active-high") << " reset" << endl;
    *logger << log_level::verbose << "Using " << clk << " as clock with half-period of " << clk_half_period << " simulation steps" << endl;
    cycle_count = 0;
}

XSI_DUT::~XSI_DUT(){
    xsi.close();
}

void XSI_DUT::run_ncycles(unsigned int n){
    for(int i=0; i<n; i++){
        write(clk, 0);
        xsi.run(clk_half_period);
        write(clk, 1);
        xsi.run(clk_half_period);
        cycle_count++;
    }
}

uint64_t XSI_DUT::get_cycle_count(){
    return cycle_count;
}

void XSI_DUT::list_ports(){
    map<string, port_parameters>::iterator it = port_map.begin();
    while(it != port_map.end()){
        *logger << log_level::verbose << it->first << " (ID: " << it->second.port_id << ", "
                            << it->second.port_bits << "b, "
                                << (it->second.is_input ? "I)" : "O)") << endl;
        it++;
    }
}

void XSI_DUT::reset_design(){
    map<string, port_parameters>::iterator it = port_map.begin();
    while(it != port_map.end()){
        if(it->second.is_input){
            write(it->first, 0);
        }
        it++;
    }
    write(rst, rst_active_low ? 0 : 1);
    run_ncycles(1000);
    write(rst, rst_active_low ? 1 : 0);
    run_ncycles(10);
}

void XSI_DUT::rewind(){
    xsi.restart();
}

int XSI_DUT::num_ports(){
    return port_map.size();
}

void XSI_DUT::write(const std::string &port_name, unsigned int val){
    if(!port_map[port_name].is_input){
        throw invalid_argument("Write called on output port");
    }
    unsigned int nwords = (port_map[port_name].port_bits+31)/32; //find how many 32-bit chunks we need
    vector<s_xsi_vlog_logicval> logic_val(nwords);
    logic_val.at(0) = (s_xsi_vlog_logicval){val, 0};
    for(int i=1; i<nwords; i++){
        logic_val.at(i) = (s_xsi_vlog_logicval){0, 0};//only two-valued logic
    }
    xsi.put_value(port_map[port_name].port_id, logic_val.data());
}

unsigned int XSI_DUT::read(const std::string &port_name){
    unsigned int nwords = (port_map[port_name].port_bits+31)/32; //find how many 32-bit chunks we need
    if(nwords > 1){
        throw invalid_argument("uint = read(string name) applies only to signals of 32b or less");
    }
    vector<s_xsi_vlog_logicval> logic_val(nwords);
    xsi.get_value(port_map[port_name].port_id, logic_val.data());
    return logic_val.at(0).aVal;
}

void XSI_DUT::set(const string &port_name){
    if(port_map[port_name].port_bits != 1){
        throw invalid_argument("set() applies only to scalars");
    }
    write(port_name, (ap_uint<1>)1);
}

void XSI_DUT::clear(const string &port_name){
    if(port_map[port_name].port_bits != 1){
        throw invalid_argument("clear() applies only to scalars");
    }
    write(port_name, (ap_uint<1>)0);
}

bool XSI_DUT::test(const string &port_name){
    if(port_map[port_name].port_bits != 1){
        throw invalid_argument("test() applies only to scalars");
    }
    ap_uint<1> ret = read(port_name);
    return (ret == 1);
}
