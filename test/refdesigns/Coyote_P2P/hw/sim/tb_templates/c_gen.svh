import lynxTypes::*;
import simTypes::*;

// AXIS Generator
class c_gen;
  
  // Send to driver (mailbox)
  mailbox gen2drv;

  // Params
  c_struct_t params;

  // Completion
  event done;

  // Stream type
  integer strm_type;
  
  //
  // C-tor
  //
  function new(mailbox gen2drv, input integer strm_type, input c_struct_t params);
    this.gen2drv = gen2drv;
    this.params = params;
    this.strm_type = strm_type;
  endfunction
  
  //
  // Run
  // --------------------------------------------------------------------------
  // This is the function to edit if any custom stimulus is needed. 
  // By default it will generate random stimulus n_trs times.
  // --------------------------------------------------------------------------
  //
  
  task run();
    c_trs trs;

    for(int i = 0; i < params.n_trs; i++) begin
      trs = new();
      if(!trs.randomize()) $fatal("ERR:  Generator randomization failed");
      trs.tlast = i == params.n_trs-1;
      trs.display("Gen");
      gen2drv.put(trs);
    end 
    -> done;
  endtask

endclass