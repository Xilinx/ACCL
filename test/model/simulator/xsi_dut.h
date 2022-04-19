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

#pragma once

#include "xsi_loader.h"
#include <map>
#include <string>
#include <cstdint>
#include "ap_int.h"
#include "log.hpp"

typedef struct{
    int port_id;
    int port_bits;
    bool is_input;
} port_parameters;

class XSI_DUT{

public:
    XSI_DUT(const std::string& design_libname, const std::string& simkernel_libname,
            const std::string& reset_name, bool reset_active_low,
            const std::string& clock_name, float clock_period_ns,
            const std::string& wdb_name, Log &log);
    ~XSI_DUT();
    void list_ports();
    int num_ports();
    void reset_design();
    void rewind();
    void run_ncycles(unsigned int n);
    template<unsigned int W> void write(const std::string &port_name, ap_uint<W> val);
    template<unsigned int W> ap_uint<W> read(const std::string &port_name);
    void write(const std::string &port_name, unsigned int val);
    unsigned int read(const std::string &port_name);
    void set(const std::string &port_name);
    void clear(const std::string &port_name);
    bool test(const std::string &port_name);
    uint64_t get_cycle_count();
private:
    //global instance of the XSI object
    Xsi::Loader xsi;
    //port map
    std::map<std::string, port_parameters> port_map;
    //names of clock and reset
    std::string clk;
    std::string rst;
    unsigned int clk_half_period;
    bool rst_active_low;
    uint64_t cycle_count;
};

template<unsigned int W>
void XSI_DUT::write(const std::string &port_name, ap_uint<W> val){
    if(W != port_map[port_name].port_bits){
        throw std::invalid_argument("Value bitwidth does not match port bitwidth");
    }
    constexpr int nwords = (W+31)/32; //find how many 32-bit chunks we need
    s_xsi_vlog_logicval logic_val[nwords];
    for(int i=0; i<nwords; i++){
        logic_val[i] = (s_xsi_vlog_logicval){(XSI_UINT32)val(std::min((unsigned int)(32*(i+1)-1), W-1), 32*i), 0};//only two-valued logic
    }
    xsi.put_value(port_map[port_name].port_id, logic_val);
}

template<unsigned int W>
ap_uint<W> XSI_DUT::read(const std::string &port_name){
    if(W != port_map[port_name].port_bits){
        throw std::invalid_argument("Return value bitwidth does not match port bitwidth");
    }
    constexpr int nwords = (W+31)/32; //find how many 32-bit chunks we need
    s_xsi_vlog_logicval logic_val[nwords];
    ap_uint<W> ret;
    xsi.get_value(port_map[port_name].port_id, logic_val);
    for(int i=0; i<nwords; i++){
        ret(std::min((unsigned int)(32*(i+1)-1), W-1), 32*i) = logic_val[i].aVal;//only two-valued logic
    }
    return ret;
}
