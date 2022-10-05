..
   comment:: SPDX-License-Identifier: Apache-2.0
   comment:: Copyright (C) 2022 Advanced Micro Devices, Inc

##################################
Using ACCL compression
##################################

In general, ACCL is datatype-agnostic as most of its function involves data movement without any actual 
interaction with the data values themselves. However, there are two exceptions to this rule:

* Elementwise operations (e.g. SUM) performed by ACCL instances on buffers during reduction-type collectives
* Datatype conversions when the source and destination of a transfer are of different data types. In this scenario we say the lower-precision buffer is compressed.

To support these elementwise operations and conversions, ACCL must be configured with a reduction plugin and coversion plugins respectively. 
Reduction plugins take two operand streams and produce one result stream of the same datatype.
Compression plugins take one operand as input and produce a result of different datatype.
The input and output datatypes are specified to these plugins by the CCLO when initiating the operation.

Example reduction and conversion plugins are provided in the ACCL repo. 
The example reduction plugin supports five data types: FP16/32/64 and INT32/64. 
The example compression plugin converts between floating-point single-precision (FP32) and half-precision (FP16). 
Together, these plugins enable five homogeneous datapath configurations, which operate on buffers of identical data types, 
and one heterogeneous cofiguration for combinations of FP32 and FP16 buffers, e.g. source buffers of 
a primitive can be FP32 and results FP16 or vice-versa.

The key fields of an ACCL datatype configuration are:

* bytes per element for the compressed and uncompressed datatype. In the case of homogeneous configurations, these are the same value.
* ratio of compressed elements to uncompressed elements, i.e. how many uncompressed buffer elements are consumed in the conversion process to produce one compressed element. For elementwise conversion e.g. FP32 to FP16, this ratio is 1. For block floating point formats, this ratio could be higher.
* whether arithmetic should be performed on the compressed data - for higher throughput - or uncompressed data - for higher precision. ACCL determines the order of conversions required to meet this specifications for each primitive and collective.
* function IDs to be provided to the plugins when performing compression, decompression, and reduction.

Notice that in the ACCL default FP32/FP16 compression configuration, arithmetic is perfomed on the lower-precision FP16 
datatype. This can be changed by editing the default arithmetic configuration in the driver code.

Since C++ does not support FP16 natively, it's difficult to demonstrate compression explicitly.
However, it can be used internally to cast FP32 data to FP16 for transmission over Ethernet, 
and cast back to FP32 at the destination, as in the following example:

.. code-block:: c++

   auto op_buf = accl.create_buffer<float>(count, dataType::float32);
   auto res_buf = accl.create_buffer<float>(count, dataType::float32);
   accl.send(*op_buf, count, 1, TAG_ANY, GLOBAL_COMM, false,
               streamFlags::NO_STREAM, dataType::float16);
   accl.recv(*res_buf, count, 0, TAG_ANY, GLOBAL_COMM, false,
               streamFlags::NO_STREAM, dataType::float16);

Notice we're passing `dataType::float16` as the final argument indicating we want FP16 compression over the wire.
All of the ACCL collectives support transparent FP16 compression, and other compression mechanisms
can be implemented by simply providing an alternative compression plugin capable of converting to the desired 
datatype, and amending the datapath configuration.