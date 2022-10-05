..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

##################################
Using streaming
##################################

The CCLO IP can be utilized to move data directly to and from user-defined FPGA kernels through streams 
rather than shared memory. In this mode of operation, data being send out into the network is pulled 
from a stream interface connected to the user kernel, while data received is pushed into another 
such stream. Similarly, streaming can be utilized on only one of the ingress or egress paths.
ACCL exposes the streaming functionality through a set of flags provided as arguments to a primitive or collective.

Here is an example where the user kernel is a simple loopback FIFO.
We receive to a stream, then send the same data out from the stream to the original sender.
Because the streaming send or receives do not actually touch memory, we provide them with dummy ACCL buffers as input.
At the end of this sequence, the data in `res_buf` is identical to `op_buf`

.. code-block:: c++

   //rank 0 sends to rank 1
   accl.send(*op_buf, count, 1, 0);
   //rank 1 receives into the kernel stream
   accl.recv(*dummy_buf, count, 0, 0, GLOBAL_COMM, false, streamFlags::RES_STREAM);
   //rank 1 sends from the kernel stream to rank 0
   accl.send(*dummy_buf, count, 0, 1, GLOBAL_COMM, false, streamFlags::OP0_STREAM);
   //rank 0 receives into memory
   accl.recv(*res_buf, count, 1, 1);

The streaming flags can be applied to any ACCL operation, however for two-input primitives,
only the first input can originate from a stream. In addition, the flags can be combined,
for example to send data from a kernel stream directly into the kernel stream of the remote peer:

.. code-block:: c++
   
   //stream to stream send from to rank 1
   accl.send(*dummy_buf, count, 1, TAG_ANY, GLOBAL_COMM, false, streamFlags::OP0_STREAM | streamFlags::RES_STREAM);
   //equivalent to
   accl.stream_put(count, 1, TAG_ANY);

Here we have two equivalent ways to send data from local stream to remote stream. The `stream_put` is simply 
an alias to the send above it. However, the put syntax better expresses the fact that this is a one-sided 
operation, i.e. no receive is required (or indeed, possible) on the remote side. 
This is because receives operate by checking RX buffers, which aren't touched by the stream-to-stream operation. 

Finally, in the tag and receive-to-stream operations, the tag takes on a special meaning. 
For `stream_put`, the tag must be greater than 8, and is passed along as side-band information with the data when it 
is handed over to the kernel stream. This side-band can be used for example to route the data to one of multiple 
user kernels connected to the same stream through a switch, or to indicate other relevant information to the 
stream endpoint. For streaming receives, the tag is also passed on as metadata, but it is also used for matching 
the receive to the send as is usual in MPI. 

