This document explains, what the state of development is at and tries to document some of the decisions made

## Structure

Consists of

- wrapper, bindings and helper functionality found in ./accl_process_group
- main C++ files in ./src
- The ACCL repo the process group itself builds on top will be in ./accl . This is replicated such that you can try different versions
- ./test testscripts

## Build process

Check the ./install.py helper for dependency versions

./setup.py sets up the build

See the section in the README on how to avoid the long build using pip

## Basics

- Currently only runs via Coyote RDMA. XRT and GPU support was dropped. Simulator still runs over XRT UDP though
- Needs MPI Library to work. Set in setup.py. Tested only with MPICH
- The test setup in run.sh is for the HACC cluster
- use ACCL_DEBUG=1 both during build and runs
- Everything runs in rendezvous mode
- if you call collectives directly they are run synchronously, but eg allreduce used internally in DDP is executed async
- The PG allocates 2 buffers and reuses them to avoid reallocation. This is supposed to be replaced with a host buffer constructor which takes an existing memory region. To change buffer type you need to use the change_buffer_type branch(maybe already pulled) at https://github.com/lawirz/ACCL 
- The torch profiler can see the overall execution time, but setting it up to measure sub-operation within the workerthread was attempted but failed.

## ProcessGroupACCL.cpp

### ProcessGroup structure

A lot of the design comes from the ProcessGroupMPI. There is a concept of WorkEntries, which schedule Work on a separate worker thread. This is currently done using a single Worker thread as is the case with the MPI PG. There is still a lock, probably only relevant in case of a few management operations from the DDP side. With async execution in ACCL, we could try a different structure with AsyncWork as is done on Gloo PG I think.

### Collectives

- There are small wrappers, which do a few checks mostly copied from MPI PG, do the sidestep then setup the WorkEntry
- The WorkEntries manage the Segmentation, which is not yet correctly implemented everywhere. Some collectives still use a version which relies on the input to have one-dimensional shape. Others, which require multiple Segmentations such as Scatter have similar limitations
- Input is copied to the pre-allocated buffer. Generally copies using memcpy seem to be much faster, than using tensor.copy_ for some reason
- ACCL does a host-to-host call. The driver figures out, that it's host to host using the buffer type. The compressed type should be added as an argument to make that work again
- copy back

## Hardware issues

A lot of collectives still fail in hardware. The following can produce issues

- Mixing datatypes especially ints
- High variablity in length
- MPI sidestepping(can't explain why this causes issues)

If you run test-resnet50, you will encounter them.
