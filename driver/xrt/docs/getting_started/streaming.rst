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
ACCL overloads primitives or collectives to let the user optionally specify input and output buffers.
If no buffers are specified with the fucntion call the data is stremed from or to the user kernel instead.

Here is an example where the user kernel is a simple loopback FIFO.
We receive to a stream, then send the same data out from the stream to the original sender.
Because the streaming send or receives do not actually touch memory, we use a slightly different function signature.
Instead of specifying a source or destination buffer, we only specify the data type. 
In that case, the data is pulled or pushed to the stram interface.
At the end of this sequence, the data in `res_buf` is identical to `op_buf`.
The data type of both buffers is `dataType::float32`.

.. code-block:: c++

   //rank 0 sends to rank 1
   accl.send(*op_buf, count, 1, 0);
   //rank 1 receives into the kernel stream. Note, that we specify the data type 
   //instead of a buffer as the first argument.
   accl.recv(dataType::float32, count, 0, 0);
   //rank 1 sends from the kernel stream to rank 0
   accl.send(dataType::float32, count, 0, 1);
   //rank 0 receives into memory
   accl.recv(*res_buf, count, 1, 1);

It exists a streaming version for every ACCL operation. However, for two-input primitives,
only the first input can originate from a stream. In addition, it is possible to send data from a kernel 
stream directly into the kernel stream of the remote peer:

.. code-block:: c++
   
   //stream to stream send from arbitrary rank to rank 1
   accl.stream_put(dataType::float32, count, 1, TAG_ANY);

The `stream_put` is a one-sided operation, i.e. no receive is required (or indeed, possible) on the remote side. 
This is because receives operate by checking RX buffers, which aren't touched by the stream-to-stream operation. 
Instead, the data is directly passed to the streaming interface connected to the user kernels.

Finally, in the tag and receive-to-stream operations, the tag takes on a special meaning. 
For `stream_put`, the tag must be greater than 8, and is passed along as side-band information with the data when it 
is handed over to the kernel stream. This side-band can be used for example to route the data to one of multiple 
user kernels connected to the same stream through a switch, or to indicate other relevant information to the 
stream endpoint. For streaming receives, the tag is also passed on as metadata, but it is also used for matching 
the receive to the send as is usual in MPI. 

