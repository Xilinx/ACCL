/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  * Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.
  * All rights reserved.
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

#ifndef __FPGA_FOPS_H__
#define __FPGA_FOPS_H__

#include "coyote_dev.h"
#include "fpga_mmu.h"

/* Pid */
int32_t register_pid(struct fpga_dev *d, pid_t pid);
int unregister_pid(struct fpga_dev *d, int32_t cpid);

/* Engine status read */
uint32_t engine_status_read(struct xdma_engine *engine);

/* Reconfiguration */
int reconfigure(struct fpga_dev *d, uint64_t vaddr, uint64_t len);
int alloc_pr_buffers(struct fpga_dev *d, unsigned long n_pages);
int free_pr_buffers(struct fpga_dev *d, uint64_t vaddr);

/* Fops */
int fpga_open(struct inode *inode, struct file *file);
int fpga_release(struct inode *inode, struct file *file);
int fpga_mmap(struct file *file, struct vm_area_struct *vma);
long fpga_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

#endif // FPGA FOPS
