/**
  * Copyright (c) 2021, Systems Group, ETH Zurich
  * All rights reserved.
  * Modifications Copyright (c) 2024, Advanced Micro Devices, Inc.
  * All rights reserved.
  *
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

#include "fpga_mmu.h"
#include "pci/pci_dev.h"
#include <linux/dma-buf.h>
#include <linux/dma-direct.h>
#include <linux/dma-resv.h>

/* User tables */
struct hlist_head user_lbuff_map[MAX_N_REGIONS][1 << (USER_HASH_TABLE_ORDER)]; // large alloc
struct hlist_head user_sbuff_map[MAX_N_REGIONS][1 << (USER_HASH_TABLE_ORDER)]; // main alloc

/* PR table */
struct hlist_head pr_buff_map[1 << (PR_HASH_TABLE_ORDER)];

/**
 * @brief ALlocate user buffers (used in systems without hugepage support)
 * 
 * @param d - vFPGA
 * @param n_pages - number of pages to allocate
 * @param cpid - Coyote PID
 */
int alloc_user_buffers(struct fpga_dev *d, unsigned long n_pages, int32_t cpid)
{
    int i, ret_val = 0;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    if (d->curr_user_buff.n_hpages) {
        dbg_info("allocated user buffers exist and are not mapped\n");
        return -1;
    }

    // check host
    if (n_pages > MAX_BUFF_NUM) 
        d->curr_user_buff.n_hpages = MAX_BUFF_NUM;
    else
        d->curr_user_buff.n_hpages = n_pages;

    // check card
    if(pd->en_mem)
        if (d->curr_user_buff.n_hpages > pd->num_free_lchunks)
            return -ENOMEM;

    d->curr_user_buff.huge = true;
    d->curr_user_buff.cpid = cpid;

    // alloc host
    d->curr_user_buff.hpages = kzalloc(d->curr_user_buff.n_hpages * sizeof(*d->curr_user_buff.hpages), GFP_KERNEL);
    if (d->curr_user_buff.hpages == NULL) {
        return -ENOMEM;
    }
    dbg_info("allocated %llu bytes for page pointer array for %lld user host buffers @0x%p.\n",
             d->curr_user_buff.n_hpages * sizeof(*d->curr_user_buff.hpages), d->curr_user_buff.n_hpages, d->curr_user_buff.hpages);

    for (i = 0; i < d->curr_user_buff.n_hpages; i++) {
        d->curr_user_buff.hpages[i] = alloc_pages(GFP_ATOMIC, pd->ltlb_order->page_shift - PAGE_SHIFT);
        if (!d->curr_user_buff.hpages[i]) {
            dbg_info("user host buffer %d could not be allocated\n", i);
            goto fail_host_alloc;
        }

        dbg_info("user host buffer allocated @ %llx device %d\n", page_to_phys(d->curr_user_buff.hpages[i]), d->id);
    }

    // alloc card
    if(pd->en_mem) {
        d->curr_user_buff.n_pages = d->curr_user_buff.n_hpages;
        d->curr_user_buff.cpages = kzalloc(d->curr_user_buff.n_pages * sizeof(uint64_t), GFP_KERNEL);
        if (d->curr_user_buff.cpages == NULL) {
            return -ENOMEM;
        }
        dbg_info("allocated %llu bytes for page pointer array for %lld user card buffers @0x%p.\n",
                d->curr_user_buff.n_pages * sizeof(*d->curr_user_buff.cpages), d->curr_user_buff.n_pages, d->curr_user_buff.cpages);

        ret_val = card_alloc(d, d->curr_user_buff.cpages, d->curr_user_buff.n_pages, LARGE_CHUNK_ALLOC);
        if (ret_val) {
            dbg_info("user card buffer %d could not be allocated\n", i);
            goto fail_card_alloc;
        }
    }

    return 0;
fail_host_alloc:
    while (i)
        __free_pages(d->curr_user_buff.hpages[--i], pd->ltlb_order->page_shift - PAGE_SHIFT);

    d->curr_user_buff.n_hpages = 0;

    kfree(d->curr_user_buff.hpages);

    return -ENOMEM;

fail_card_alloc:
    // release host
    for (i = 0; i < d->curr_user_buff.n_hpages; i++)
        __free_pages(d->curr_user_buff.hpages[i], pd->ltlb_order->page_shift - PAGE_SHIFT);

    d->curr_user_buff.n_hpages = 0;
    d->curr_user_buff.n_pages = 0;

    kfree(d->curr_user_buff.hpages);
    kfree(d->curr_user_buff.cpages);

    return -ENOMEM;
}

/**
 * @brief Free user buffers
 * 
 * @param d - vFPGA
 * @param vaddr - virtual address 
 * @param cpid - Coyote PID
 */
int free_user_buffers(struct fpga_dev *d, uint64_t vaddr, int32_t cpid)
{
    int i;
    uint64_t vaddr_tmp;
    struct user_pages *tmp_buff;  
    uint64_t *map_array;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    hash_for_each_possible(user_lbuff_map[d->id], tmp_buff, entry, vaddr) {

        if (tmp_buff->vaddr == vaddr && tmp_buff->cpid == cpid) {

            vaddr_tmp = tmp_buff->vaddr;

            // free host pages
            for (i = 0; i < tmp_buff->n_hpages; i++) {
                if (tmp_buff->hpages[i])
                    __free_pages(tmp_buff->hpages[i], pd->ltlb_order->page_shift - PAGE_SHIFT);
            }
            kfree(tmp_buff->hpages);

            // free card pages
            if(pd->en_mem) {
                card_free(d, tmp_buff->cpages, tmp_buff->n_pages, LARGE_CHUNK_ALLOC);
                kfree(tmp_buff->cpages);
            }

            // map array
            map_array = (uint64_t *)kzalloc(tmp_buff->n_hpages * 2 * sizeof(uint64_t), GFP_KERNEL);
            if (map_array == NULL) {
                dbg_info("map buffers could not be allocated\n");
                return -ENOMEM;
            }

            // fill mappings
            for (i = 0; i < tmp_buff->n_hpages; i++) {
                tlb_create_unmap(pd->ltlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                vaddr_tmp += pd->ltlb_order->page_size;
            }

            // fire
            tlb_service_dev(d, pd->ltlb_order, map_array, tmp_buff->n_hpages);

            // free
            kfree((void *)map_array);

            // Free from hash
            hash_del(&tmp_buff->entry);
        }
    }

    return 0;
}

 /**
 * @brief Allocate PR buffers
 * 
 * @param d - vFPGA
 * @param n_pages - number of pages to allocate
 */
int alloc_pr_buffers(struct fpga_dev *d, unsigned long n_pages)
{
    int i;
    struct pr_ctrl *prc;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    prc = d->prc;
    BUG_ON(!prc);
    pd = d->pd;
    BUG_ON(!pd);

    // obtain PR lock
    spin_lock(&prc->lock);

    if (prc->curr_buff.n_pages) {
        dbg_info("allocated PR buffers exist and are not mapped\n");
        return -1;
    }

    if (n_pages > MAX_PR_BUFF_NUM)
        prc->curr_buff.n_pages = MAX_PR_BUFF_NUM;
    else
        prc->curr_buff.n_pages = n_pages;

    prc->curr_buff.pages = kzalloc(n_pages * sizeof(*prc->curr_buff.pages), GFP_KERNEL);
    if (prc->curr_buff.pages == NULL) {
        return -ENOMEM;
    }

    dbg_info("allocated %lu bytes for page pointer array for %ld PR buffers @0x%p.\n",
             n_pages * sizeof(*prc->curr_buff.pages), n_pages, prc->curr_buff.pages);

    for (i = 0; i < prc->curr_buff.n_pages; i++) {
        prc->curr_buff.pages[i] = alloc_pages(GFP_ATOMIC, pd->ltlb_order->page_shift - PAGE_SHIFT);
        if (!prc->curr_buff.pages[i]) {
            dbg_info("PR buffer %d could not be allocated\n", i);
            goto fail_alloc;
        }

        dbg_info("PR buffer allocated @ %llx \n", page_to_phys(prc->curr_buff.pages[i]));
    }

    // release PR lock
    spin_unlock(&prc->lock);

    return 0;
fail_alloc:
    while (i)
        __free_pages(prc->curr_buff.pages[--i], pd->ltlb_order->page_shift - PAGE_SHIFT);
    // release PR lock
    spin_unlock(&prc->lock);
    return -ENOMEM;
}

/**
 * @brief Free PR pages
 * 
 * @param d - vFPGA
 * @param vaddr - virtual address
 */
int free_pr_buffers(struct fpga_dev *d, uint64_t vaddr)
{
    int i;
    struct pr_pages *tmp_buff;
    struct pr_ctrl *prc;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    prc = d->prc;   
    BUG_ON(!prc);
    pd = d->pd;
    BUG_ON(!pd);

    // obtain PR lock
    spin_lock(&prc->lock);

    hash_for_each_possible(pr_buff_map, tmp_buff, entry, vaddr) {
        if (tmp_buff->vaddr == vaddr && tmp_buff->reg_id == d->id) {

            // free pages
            for (i = 0; i < tmp_buff->n_pages; i++) {
                if (tmp_buff->pages[i])
                    __free_pages(tmp_buff->pages[i], pd->ltlb_order->page_shift - PAGE_SHIFT);
            }

            kfree(tmp_buff->pages);

            // Free from hash
            hash_del(&tmp_buff->entry);
        }
    }

    // obtain PR lock
    spin_unlock(&prc->lock);

    return 0;
}

/**
 * @brief Allocate card memory
 * 
 * @param d - vFPGA
 * @param card_paddr - card physical address 
 * @param n_pages - number of pages to allocate
 * @param type - page size
 */
int card_alloc(struct fpga_dev *d, uint64_t *card_paddr, uint64_t n_pages, int type)
{
    int i;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;

    switch (type) {
    case 0: //
        // lock
        spin_lock(&pd->card_s_lock);

        if(pd->num_free_schunks < n_pages) {
            dbg_info("not enough free small card pages\n");
            return -ENOMEM;
        }

        for(i = 0; i < n_pages; i++) {
            pd->salloc->used = true;
            card_paddr[i] = pd->salloc->id << STLB_PAGE_BITS;
            dbg_info("user card buffer allocated @ %llx device %d\n", card_paddr[i], d->id);
            pd->salloc = pd->salloc->next;
        }

        // release lock
        spin_unlock(&pd->card_s_lock);

        break;
    case 1:
        // lock
        spin_lock(&pd->card_l_lock);

        if(pd->num_free_lchunks < n_pages) {
            dbg_info("not enough free large card pages\n");
            return -ENOMEM;
        }

        for(i = 0; i < n_pages; i++) {
            pd->lalloc->used = true;
            card_paddr[i] = (pd->lalloc->id << LTLB_PAGE_BITS) + MEM_SEP;
            dbg_info("user card buffer allocated @ %llx device %d\n", card_paddr[i], d->id);
            pd->lalloc = pd->lalloc->next;
        }

        // release lock
        spin_unlock(&pd->card_l_lock);

        break;
    default: // TODO: Shared mem
        break;
    }

    return 0;
}

/**
 * @brief Free card memory
 * 
 * @param d - vFPGA
 * @param card_paddr - card physical address 
 * @param n_pages - number of pages to free
 * @param type - page size
 */
void card_free(struct fpga_dev *d, uint64_t *card_paddr, uint64_t n_pages, int type)
{
    int i;
    uint64_t tmp_id;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    switch (type) {
    case 0: // small pages
        // lock
        spin_lock(&pd->card_s_lock);
        for(i = n_pages - 1; i >= 0; i--) {
            tmp_id = card_paddr[i] >> STLB_PAGE_BITS;
            if(pd->schunks[tmp_id].used) {
                pd->schunks[tmp_id].next = pd->salloc;
                pd->salloc = &pd->schunks[tmp_id];
            }
        }

        // release lock
        spin_unlock(&pd->card_s_lock);

        break;
    case 1: // large pages
        // lock
        spin_lock(&pd->card_l_lock);

        for(i = n_pages - 1; i >= 0; i--) {
            tmp_id = (card_paddr[i] - MEM_SEP) >> LTLB_PAGE_BITS;
            if(pd->lchunks[tmp_id].used) {
                pd->lchunks[tmp_id].next = pd->lalloc;
                pd->lalloc = &pd->lchunks[tmp_id];
            }
        }

        // release lock
        spin_unlock(&pd->card_l_lock);

        break;
    default:
        break;
    }
}

 /**
 * @brief Page map list
 * 
 * @param vaddr - starting vaddr
 * @param paddr_host - host physical address
 * @param paddr_card - card physical address
 * @param cpid - Coyote PID
 * @param entry - liste entry
 */
void tlb_create_map(struct tlb_order *tlb_ord, uint64_t vaddr, uint64_t paddr_host, uint64_t paddr_card, int32_t cpid, uint64_t *entry)
{
    uint64_t key;
    uint64_t tag;
    uint64_t phost;
    uint64_t pcard;

    key = (vaddr >> tlb_ord->page_shift) & tlb_ord->key_mask;
    tag = vaddr >> (tlb_ord->page_shift + tlb_ord->key_size);
    phost = (paddr_host >> tlb_ord->page_shift) & (uint64_t) (~0ULL);//tlb_ord->phy_mask; //To support 44-bit DMA addresses. See TODOs in fpga_dev.c/read_static_config
    pcard = (paddr_card >> tlb_ord->page_shift) & tlb_ord->phy_mask;

    // new entry
    entry[0] |= key | 
                (tag << tlb_ord->key_size) | 
                ((uint64_t)cpid << (tlb_ord->key_size + tlb_ord->tag_size)) | 
                (1UL << (tlb_ord->key_size + tlb_ord->tag_size + PID_SIZE));
    entry[1] |= phost | (pcard << tlb_ord->phy_size);

    dbg_info("creating new TLB entry, vaddr %llx, phost %llx, pcard %llx, cpid %d, hugepage %d\n", vaddr, paddr_host, paddr_card, cpid, tlb_ord->hugepage);
}

/**
 * @brief Page unmap lists
 * 
 * @param vaddr - starting vaddr
 * @param cpid - Coyote PID
 * @param entry - list entry
 */
void tlb_create_unmap(struct tlb_order *tlb_ord, uint64_t vaddr, int32_t cpid, uint64_t *entry)
{
    uint64_t tag;
    uint64_t key;

    key = (vaddr >> tlb_ord->page_shift) & tlb_ord->key_mask;
    tag = vaddr >> (tlb_ord->page_shift + tlb_ord->key_size);

    // entry host
    entry[0] |= key | 
                (tag << tlb_ord->key_size) | 
                ((uint64_t)cpid << (tlb_ord->key_size + tlb_ord->tag_size)) | 
                (0UL << (tlb_ord->key_size + tlb_ord->tag_size + PID_SIZE));
    entry[1] |= 0;

    dbg_info("unmapping TLB entry, vaddr %llx, cpid %d, hugepage %d\n", vaddr, cpid, tlb_ord->hugepage);
}

/**
 * @brief Map TLB
 * 
 * @param d - vFPGA
 * @param en_tlbf - TLBF enabled
 * @param map_array - prepped map array
 * @param paddr - physical address
 * @param cpid - Coyote PID
 * @param card - map card mem as well
 */
void tlb_service_dev(struct fpga_dev *d, struct tlb_order *tlb_ord, uint64_t* map_array, uint32_t n_pages)
{
    int i = 0;
    struct bus_drvdata *pd;

    BUG_ON(!d); 
    pd = d->pd;
    BUG_ON(!pd);

    if(pd->en_tlbf && (n_pages > MAX_MAP_AXIL_PAGES)) {
        // lock
        spin_lock(&pd->tlb_lock);

        // start DMA
        pd->fpga_stat_cnfg->tlb_addr = virt_to_phys((void *)map_array); //notice: virt_to_phys() may not work in case HW IOMMU is enabled in the PCIe infrastructure
        pd->fpga_stat_cnfg->tlb_len = n_pages * 2 * sizeof(uint64_t);
        if(tlb_ord->hugepage) {
            pd->fpga_stat_cnfg->tlb_ctrl = TLBF_CTRL_START | ((d->id && TLBF_CTRL_ID_MASK) << TLBF_CTRL_ID_SHFT);
        } else {
            pd->fpga_stat_cnfg->tlb_ctrl = TLBF_CTRL_START | (((pd->n_fpga_reg + d->id) && TLBF_CTRL_ID_MASK) << TLBF_CTRL_ID_SHFT);
        }

        // poll
        while ((pd->fpga_stat_cnfg->tlb_stat & TLBF_STAT_DONE) != 0x1)
            ndelay(100);
        
        // unlock
        spin_unlock(&pd->tlb_lock);
    } else {
        // map each page through AXIL
        for (i = 0; i < n_pages; i++) {
            if(tlb_ord->hugepage) {
                d->fpga_lTlb[0] = map_array[2*i+0];
                d->fpga_lTlb[1] = map_array[2*i+1];
            } else {
                d->fpga_sTlb[0] = map_array[2*i+0];
                d->fpga_sTlb[1] = map_array[2*i+1];
            }
        }
    }
}

/**
 * @brief Release all remaining user pages
 * 
 * @param d - vFPGA
 * @param dirtied - modified
 */
int tlb_put_user_pages_all(struct fpga_dev *d, int dirtied)
{
    int i, bkt;
    struct user_pages *tmp_buff;
    uint64_t vaddr_tmp;
    int32_t cpid_tmp;
    uint64_t *map_array;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    struct device *dev = &d->pd->pci_dev->dev;

    hash_for_each(user_sbuff_map[d->id], bkt, tmp_buff, entry) {
        
        // release host pages
        if(dirtied)
            for(i = 0; i < tmp_buff->n_hpages; i++)
                SetPageDirty(tmp_buff->hpages[i]);

        for(i = 0; i < tmp_buff->n_hpages; i++)
            put_page(tmp_buff->hpages[i]);

        kfree(tmp_buff->hpages);

        // release card pages
        if(pd->en_mem) {
            if(tmp_buff->huge)
                card_free(d, tmp_buff->cpages, tmp_buff->n_pages, LARGE_CHUNK_ALLOC);
            else
                card_free(d, tmp_buff->cpages, tmp_buff->n_pages, SMALL_CHUNK_ALLOC);
        }

        // unmap from TLB
        vaddr_tmp = tmp_buff->vaddr;
        cpid_tmp = tmp_buff->cpid;

        // map array
        map_array = (uint64_t *)kzalloc(tmp_buff->n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
        if (map_array == NULL) {
            dbg_info("map buffers could not be allocated\n");
            return -ENOMEM;
        }

        // huge pages
        if(tmp_buff->huge) {
            // fill mappings
            for (i = 0; i < tmp_buff->n_pages; i++) {
                dma_unmap_page(dev, tmp_buff->hpages[i], 2*1024*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                tlb_create_unmap(pd->ltlb_order, vaddr_tmp, cpid_tmp, &map_array[2*i]);
                vaddr_tmp += pd->ltlb_order->page_size;
            }

            // fire
            tlb_service_dev(d, pd->ltlb_order, map_array, tmp_buff->n_pages);

        // small pages
        } else {
            // fill mappings
            for (i = 0; i < tmp_buff->n_pages; i++) {
                dma_unmap_page(dev, tmp_buff->hpages[i], 4*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                tlb_create_unmap(pd->stlb_order, vaddr_tmp, cpid_tmp, &map_array[2*i]);
                vaddr_tmp += PAGE_SIZE;
            }

            // fire
            tlb_service_dev(d, pd->stlb_order, map_array, tmp_buff->n_pages);
        }

        // free
        kfree((void *)map_array);

        // remove from map
        hash_del(&tmp_buff->entry);
    }

    return 0;
}

/**
 * @brief Release user pages (cpid)
 * 
 * @param d - vFPGA
 * @param cpid - Coyote PID
 * @param dirtied - modified
 */
int tlb_put_user_pages_cpid(struct fpga_dev *d, int32_t cpid, int dirtied)
{
    int i, bkt;
    struct user_pages *tmp_buff;
    uint64_t vaddr_tmp;
    uint64_t *map_array;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    struct device *dev = &d->pd->pci_dev->dev;

    hash_for_each(user_sbuff_map[d->id], bkt, tmp_buff, entry) {
        if(tmp_buff->cpid == cpid) {
            if(tmp_buff->hpages) {
                // release host pages
                if(dirtied)
                    for(i = 0; i < tmp_buff->n_hpages; i++)
                        SetPageDirty(tmp_buff->hpages[i]);

                for(i = 0; i < tmp_buff->n_hpages; i++)
                    put_page(tmp_buff->hpages[i]);
                kfree(tmp_buff->hpages);
            }

            // release card pages
            if(pd->en_mem) {
                if(tmp_buff->huge) {
                    dbg_info("card free for large chunk");
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, LARGE_CHUNK_ALLOC);
                } else {
                    dbg_info("card free for small chunk");
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, SMALL_CHUNK_ALLOC);
                }
            }

            // unmap from TLB
            vaddr_tmp = tmp_buff->vaddr;

            // map array
            map_array = (uint64_t *)kzalloc(tmp_buff->n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
            if (map_array == NULL) {
                dbg_info("map buffers could not be allocated\n");
                return -ENOMEM;
            }

            // huge pages
            if(tmp_buff->huge) {
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    dma_unmap_page(dev, tmp_buff->hpages[i], 2*1024*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                    tlb_create_unmap(pd->ltlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += pd->ltlb_order->page_size;
                }

                // fire
                tlb_service_dev(d, pd->ltlb_order, map_array, tmp_buff->n_pages);

            // small pages
            } else {
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    dma_unmap_page(dev, tmp_buff->hpages[i], 4*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                    tlb_create_unmap(pd->stlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += PAGE_SIZE;
                }

                // fire
                tlb_service_dev(d, pd->stlb_order, map_array, tmp_buff->n_pages);
            }

            // free
            kfree((void *)map_array);

            // remove from map
            hash_del(&tmp_buff->entry);
        }
    }

    return 0;
}

/**
 * @brief Delete TLB entry, given vaddr
 * 
 * @param d - vFPGA
 * @param vaddr - starting vaddr
 * @param cpid - Coyote PID
 * @param dirtied - modified
 */
int tlb_delete_entry(struct fpga_dev *d, uint64_t vaddr, int32_t cpid, int dirtied, bool huge)
{
    uint64_t *map_array;
    struct bus_drvdata *pd;
    struct user_pages *tmp_buff;
    uint64_t vaddr_tmp;
    int i;
    
    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    hash_for_each_possible(user_sbuff_map[d->id], tmp_buff, entry, vaddr) {
        if(tmp_buff->vaddr == vaddr && tmp_buff->cpid == cpid) {
            // // release card pages 
            
            //kfree(tmp_buff->hpages);

            if(pd->en_mem) {
                if(tmp_buff->huge) {
                    dbg_info("card free for large chunk");
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, LARGE_CHUNK_ALLOC);
                } else {
                    dbg_info("card free for small chunk");
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, SMALL_CHUNK_ALLOC); 
                }
            }

            vaddr_tmp = tmp_buff->vaddr;


            map_array = (uint64_t *)kzalloc(tmp_buff->n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
            if (map_array == NULL) {
                dbg_info("map buffers could not be allocated\n");
                return -ENOMEM;
            }

            // huge pages
            if(tmp_buff->huge) {
                dbg_info("deleting 2MB entry");
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    tlb_create_unmap(pd->ltlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += pd->ltlb_order->page_size;
                }

                // fire
                tlb_service_dev(d, pd->ltlb_order, map_array, tmp_buff->n_pages);
            } else {
                dbg_info("deleting 4kB entry");
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    tlb_create_unmap(pd->stlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += PAGE_SIZE;
                }

                // fire
                tlb_service_dev(d, pd->stlb_order, map_array, tmp_buff->n_pages);
            }
            

            // free
            kfree((void *)map_array);

            // remove from map
            hash_del(&tmp_buff->entry);
        }
    }
    return 0;
}

/**
 * @brief Release user pages
 * 
 * @param d - vFPGA
 * @param vaddr - starting vaddr
 * @param cpid - Coyote PID
 * @param dirtied - modified
 */
int tlb_put_user_pages(struct fpga_dev *d, uint64_t vaddr, int32_t cpid, int dirtied)
{
    int i;
    struct user_pages *tmp_buff;
    uint64_t vaddr_tmp;
    uint64_t *map_array;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    struct device *dev = &d->pd->pci_dev->dev;

    hash_for_each_possible(user_sbuff_map[d->id], tmp_buff, entry, vaddr) {
        if(tmp_buff->vaddr == vaddr && tmp_buff->cpid == cpid) {

            // release host pages
            if(dirtied)
                for(i = 0; i < tmp_buff->n_hpages; i++)
                    SetPageDirty(tmp_buff->hpages[i]);

            for(i = 0; i < tmp_buff->n_hpages; i++)
                put_page(tmp_buff->hpages[i]);

            kfree(tmp_buff->hpages);

            // release card pages
            if(pd->en_mem) {
                if(tmp_buff->huge)
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, LARGE_CHUNK_ALLOC);
                else
                    card_free(d, tmp_buff->cpages, tmp_buff->n_pages, SMALL_CHUNK_ALLOC);
            }

            // unmap from TLB
            vaddr_tmp = vaddr;

            // map array
            map_array = (uint64_t *)kzalloc(tmp_buff->n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
            if (map_array == NULL) {
                dbg_info("map buffers could not be allocated\n");
                return -ENOMEM;
            }

            // huge pages
            if(tmp_buff->huge) {
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    if(!tmp_buff->hpages[i])
                        continue;
                    dbg_info("unmapping huge %lx\n", tmp_buff->hpages[i]);
                    dma_unmap_page(dev, tmp_buff->hpages[i], 2*1024*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                    tlb_create_unmap(pd->ltlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += pd->ltlb_order->page_size;
                }

                // fire
                tlb_service_dev(d, pd->ltlb_order, map_array, tmp_buff->n_pages);

            // small pages
            } else {
                // fill mappings
                for (i = 0; i < tmp_buff->n_pages; i++) {
                    if(!tmp_buff->hpages[i])
                        continue;
                    dbg_info("unmapping small %lx\n", tmp_buff->hpages[i]);
                    dma_unmap_page(dev, tmp_buff->hpages[i], 4*1024, DMA_BIDIRECTIONAL); //For IOMMU management. KNOWN ISSUE: dma_unmap_page crashes on a kernel WARN_ON
                    tlb_create_unmap(pd->stlb_order, vaddr_tmp, cpid, &map_array[2*i]);
                    vaddr_tmp += PAGE_SIZE;
                }

                // fire
                tlb_service_dev(d, pd->stlb_order, map_array, tmp_buff->n_pages);
            }

            // free
            kfree((void *)map_array);

            // remove from map
            hash_del(&tmp_buff->entry);
        }
    }

    return 0;
}

/** 
* 
 * This struct keeps track of the imported DMABufs, to manage their release at the end of their usage.
 * Failure in releasing DMABufs may prevent sysadmin from `sudo rmmod` Coyote driver.
 * 
 */
typedef struct dma_buf_context {
    int fd;
    struct dma_buf * buf;
    struct dma_buf_attachment *dma_attach;
    struct sg_table *sgt;
    void * vaddr;
    uint32_t offset;
    struct dma_buf_context * next;
} dma_buf_context;

//Linked list to keep track of the imported DMABufs from the GPU driver
dma_buf_context * gpu_buffers_dma_buf_list = NULL;

/**
 * @brief Creates a new DMABuf context
 * 
 * @param list - linked list into which to add the context
 * @param fd - DMABuf file descriptor
 * @param dma_attach - dma_buf_attachment produced during importing
 * @param sgt - scattergather table received during importing
 * @param vaddr - virtual address of the buffer
 * @param offset - offset of the buffer in physical memory pages
 */ 
struct dma_buf_context * create_dma_buf_context(dma_buf_context ** list, int fd, struct dma_buf * buf, struct dma_buf_attachment *dma_attach, struct sg_table *sgt, void * vaddr, uint32_t offset) {

    dma_buf_context *current_dma_buf = *list;
    dma_buf_context *prev = NULL;

    printk(KERN_INFO "*list = %lx\n", *list);

    while(current_dma_buf != NULL) {
        prev = current_dma_buf;
        current_dma_buf = current_dma_buf->next;
    }

    current_dma_buf = (dma_buf_context *)kzalloc(sizeof(dma_buf_context), GFP_KERNEL);
    
    if(current_dma_buf == NULL) {
        pr_err("Error in DMABuf allocation.");
        return NULL;
    }
    if(prev != NULL) {
        prev->next = current_dma_buf;
    } else {
        *list = current_dma_buf;
    }

    current_dma_buf->fd = fd;
    current_dma_buf->buf = buf;
    current_dma_buf->dma_attach = dma_attach;
    current_dma_buf->sgt = sgt;
    current_dma_buf->vaddr = vaddr;
    current_dma_buf->offset = offset;
    current_dma_buf->next = NULL;

    dbg_info("fd = %d, buf = %llx, dma_attach = %llx, sgt = %llx, vaddr = %llx, offset = %llx", fd, buf, dma_attach, sgt, vaddr, offset);

    return current_dma_buf;
}

/**
 * @brief retrieve DMABuf context given its file descriptor
 * 
 * @param list - linked list into which to search
 * @param fd - DMABuf file descriptor
 */
dma_buf_context * get_dma_buf_context(dma_buf_context * list, int fd) {
    dma_buf_context * current_dma_buf = list;
    printk(KERN_INFO "list = %lx\n", list);
    while(current_dma_buf != NULL) {
        printk(KERN_INFO "get_dma_buf_context(): visiting fd = %d\n", current_dma_buf->fd);
        if(current_dma_buf->fd == fd) {
            return current_dma_buf;
        }
        current_dma_buf = current_dma_buf->next;
    }
    
    return NULL;
}

/**
 * @brief retrieve DMABuf context given its dma_buf * pointer.
 * 
 * @param list - linked list into which to search 
 * @param buf - pointer to struct dma_buf
 */
dma_buf_context * get_dma_buf_context_by_ptr(dma_buf_context * list, struct dma_buf * buf) {
    dma_buf_context * current_dma_buf = list;
    while(current_dma_buf != NULL) {
        if(current_dma_buf->buf == buf) {
            return current_dma_buf;
        }
        current_dma_buf = current_dma_buf->next;
    }
    
    return NULL;
}

/**
 * @brief Delete a DMABuf context, given its file descriptor
 * 
 * @param list - linked list into which to search
 * @param fd - DMABuf file descriptor
 */ 
void delete_dma_buf_context(dma_buf_context ** list, int fd) {
    dma_buf_context * current_dma_buf = *list;
    dma_buf_context * prev = NULL;
    printk(KERN_INFO "*list = %lx\n", *list);
    while(current_dma_buf != NULL && current_dma_buf->fd != fd) {
        prev = current_dma_buf;
        current_dma_buf = current_dma_buf->next;
    }
    if(current_dma_buf != NULL) {
        if(prev != NULL) {
            prev->next = current_dma_buf->next;
        } else {
            *list = current_dma_buf->next;
        }
        dbg_info("deleting %llx", current_dma_buf);
        kfree((void *)current_dma_buf);
    }
}

/** 
 * @brief Detaches all DMABufs imported from the GPU in case of internal errors, when cProcess instance is closed
 */
void delete_all_gpu_contexts() {
    dma_buf_context * list = gpu_buffers_dma_buf_list;
    while(list != NULL) {
        dbg_info("releasing fd %d", list->fd);
        kfree(list->dma_attach->importer_priv);
        dma_buf_detach(list->buf, list->dma_attach);
        dma_buf_put(list->buf);
        delete_dma_buf_context(&gpu_buffers_dma_buf_list, list->fd);
    }
}

//Required by p2p_move_notify()
struct gpu_move_notify_private {
    int cpid;
    int pid;
    int dirtied;
    struct fpga_dev * d;
};


//TODO: insert a timer between entries deletion and re-mapping. For lack of documentation, it is not clear
//how long it is required to wait before the TLB remapping stage.
/**
 * @brief move_notify callback for the DMABuf dynamic importer. 
 * 
 * To manage page movements in GPU memory, this routines deletes TLB entries and retrieves new entries
 * 
 * @param attach - provided by the exporter to delete old TLB entries
*/
void p2p_move_notify(struct dma_buf_attachment *attach) {
    dma_buf_context * context = get_dma_buf_context_by_ptr(gpu_buffers_dma_buf_list, attach->dmabuf);
    if(context == NULL) {
        pr_err("Error: invalid dmabuf.");
        return;
    }

    dbg_info("dma_buf: %llx, dma_attach: %llx, sgt: %llx, vaddr: %llx\n", context->buf, context->dma_attach, context->sgt, context->vaddr);
    

    int i = 0;
    void * temp_vaddr = context->vaddr;

    struct gpu_move_notify_private * importer_priv = (struct gpu_move_notify_private *) attach->importer_priv;

    int j, k;
    void * temp_vaddr2 = temp_vaddr;
    struct scatterlist * tmp_sgl, *tmp2_sgl;
    struct scatterlist * sgl = context->sgt->sgl;

    //retrieve coalesced TLB entries
    do {
        if(sg_dma_address(sgl) == 0xffffffffffffffff) { //amdgpu driver bug
            sgl = sg_next(sgl);
            continue;
        }
        unsigned int tmp = sg_dma_len(sgl);
        int num_entries = 1;
        uint64_t dma_addr = sg_dma_address(sgl) + context->offset;
        if(tmp < 2*1024*1024) {
            tmp_sgl = sg_next(sgl);
            while (tmp < 2 *1024 * 1024) {
                if(sg_dma_address(tmp_sgl) + context->offset == dma_addr + tmp) {
                    tmp += sg_dma_len(tmp_sgl);
                    num_entries++;
                    tmp_sgl = sg_next(tmp_sgl);
                } else {
                    break;
                }
            }
        }

        //delete 2MB entries
        dbg_info("deleting 2MB entries\n");
        while(tmp / (2 * 1024 * 1024) > 0) {
            tlb_delete_entry(importer_priv->d, temp_vaddr, importer_priv->cpid, importer_priv->dirtied, true);
            tmp -= 2 * 1024 * 1024;
            dma_addr += 2 * 1024 * 1024;
            temp_vaddr += 2 * 1024 * 1024;
        }

        //delete 4kB entries
        dbg_info("deleting 4kB entries\n");
        while(tmp / (4 * 1024) > 0) {
            tlb_delete_entry(importer_priv->d, temp_vaddr, importer_priv->cpid, importer_priv->dirtied, false);
            tmp -= 4 * 1024;
            dma_addr += 4 * 1024;
            temp_vaddr += 4 * 1024;
        }

        //ignore coalesced entries
        for(j = 0; j < num_entries; j++) {
            sgl = sg_next(sgl);
        }
    } while(sgl);
    
    //unmap buffer from FPGA bus address space
    dma_buf_unmap_attachment(context->dma_attach, context->sgt, DMA_BIDIRECTIONAL);
    temp_vaddr = context->vaddr;
    temp_vaddr2 = temp_vaddr;
    j = 0;
    k = 0;

    //TODO: insert timer here, to make sure that the new mapping is available from the driver.
    //Lack of docs regarding the amount of time to wait for!

    //re-map buffer into FPGA bus address space
    context->sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
    sgl = context->sgt->sgl;
    
    //coalesce entries based on contiguous memory regions
    do {
        if(sg_dma_address(sgl) == 0xffffffffffffffff) { //amdgpu driver bug
            sgl = sg_next(sgl);
            continue;
        }
        unsigned int tmp = sg_dma_len(sgl);
        int num_entries = 1;
        uint64_t dma_addr = sg_dma_address(sgl) + context->offset;
        if(tmp < 2*1024*1024) {
            tmp_sgl = sg_next(sgl);
            while (tmp < 2 *1024 * 1024) {
                if(sg_dma_address(tmp_sgl) + context->offset == dma_addr + tmp) {
                    tmp += sg_dma_len(tmp_sgl);
                    num_entries++;
                    tmp_sgl = sg_next(tmp_sgl);
                } else {
                    break;
                }
            }
        }

        dbg_info("contiguous region found: dma_addr: 0x%lx, length: 0x%lx, num_entries: %d\n", dma_addr, tmp, num_entries);

        //2MB entries
        while(tmp / (2 * 1024 * 1024) > 0) {
            dbg_info("huge entry VADDR: 0x%llx, DMA_ADDR: 0x%llx, size: 0x%llx, offset: %u\n", temp_vaddr, dma_addr, 2*1024*1024, context->offset);
            tlb_set_entry(importer_priv->d, temp_vaddr, dma_addr, importer_priv->cpid, importer_priv->pid, true);
            tmp -= 2 * 1024 * 1024;
            dma_addr += 2 * 1024 * 1024;
            temp_vaddr += 2 * 1024 * 1024;
        }

        //4kB entries
        while(tmp / (4 * 1024) > 0) {
            dbg_info("small entry VADDR: 0x%llx, DMA_ADDR: 0x%llx, size: 0x%llx, offset: %u\n", temp_vaddr, dma_addr, 4*1024, context->offset);
            tlb_set_entry(importer_priv->d, temp_vaddr, dma_addr, importer_priv->cpid, importer_priv->pid, false);
            tmp -= 4 * 1024;
            dma_addr += 4 * 1024;
            temp_vaddr += 4 * 1024;
        }

        //ignore coalesced entries
        for(j = 0; j < num_entries; j++) {
            sgl = sg_next(sgl);
        }
    } while(sgl);

    context->buf = attach->dmabuf;
    context->dma_attach = attach;

    dbg_info("Exiting from dma_buf_unmap_attachment: Here I have to print the state");

}

//required by dma_buf_dynamic_attach in p2p_attach_dma_buf
const struct dma_buf_attach_ops gpu_importer_ops = {
    .allow_peer2peer = true,
    .move_notify = p2p_move_notify 
};

/**
 * @brief Attach to a given DMABuf, for peer-to-peer DMA
 * 
 * @param d - the vFPGA
 * @param buf_fd - the file descriptor of the DMABuf, given by the exporter
 * @param vaddr - the virtual address of the buffer
 * @param offset - the offset of the buffer in the physical memory pages, given the exporter
 * @param cpid - Coyote PID
 * @param pid - User PID
 */ 
int p2p_attach_dma_buf(struct fpga_dev *d, uint64_t buf_fd, void * vaddr, uint32_t offset, int32_t cpid, pid_t pid) {

    struct device *dev = &d->pd->pci_dev->dev;
    int rc = 0;

    int err = 0;
    
    int dma_buf_fd = buf_fd;

    //retrieve dmabuf
    struct dma_buf * buf = dma_buf_get(dma_buf_fd);

    if(IS_ERR(buf)) {
        pr_err("ERROR: dma_buf_get failed.\n");
        return -EINVAL;
    }

    //create dmabuf context
    struct dma_buf_context * context = create_dma_buf_context(&gpu_buffers_dma_buf_list, dma_buf_fd, buf, NULL, NULL, vaddr, offset);
    if(context == NULL) {
        pr_err("Error in dma_buf_context creation!");
        return -ENOMEM;
    }
    
    struct gpu_move_notify_private * importer_priv = kzalloc(sizeof(struct gpu_move_notify_private), GFP_KERNEL);
    if(importer_priv == NULL) {
        pr_err("Error in importer_priv creation!");
        return -ENOMEM;
    }
    importer_priv->cpid = cpid;
    importer_priv->pid = pid;
    importer_priv->dirtied = 1;
    importer_priv->d = d;

    //attach FPGA device as a dynamic importer, to avoid data migration issues
    context->dma_attach = dma_buf_dynamic_attach(context->buf, dev, &gpu_importer_ops, importer_priv);
    
    if(IS_ERR(context->dma_attach)) {
        pr_err("ERROR: dma_buf_attach failed.\n");
        dma_buf_put(buf);
        return -EINVAL; 
    }
    
    //map p2p buffer into FPGA bus address space
    dma_resv_lock(context->buf->resv, NULL);
    context->sgt = dma_buf_map_attachment(context->dma_attach, DMA_BIDIRECTIONAL);
    dma_resv_unlock(context->buf->resv);

    if(IS_ERR(context->sgt)) {
        pr_err("ERROR: sg_table is NULL.\n");
        kfree(context->dma_attach->importer_priv);
        dma_buf_detach(context->buf, context->dma_attach);
        dma_buf_put(context->buf);
        return -EINVAL;
    }
    
    struct scatterlist *sgl = context->sgt->sgl;
    if(sgl == NULL) {
        pr_err("ERROR: scatterlist is NULL.\n");
        return -EINVAL;
    }
   
   
    //coalesce TLB entries, to produce 4kB and 2MB granularity
    int i = 0;
    int j, k;
    void * temp_vaddr = vaddr;
    void * temp_vaddr2 = temp_vaddr;
    struct scatterlist * tmp_sgl, *tmp2_sgl;
    tmp2_sgl = sgl;

    dbg_info("initial content of the scatterlist:");
    
    while(tmp2_sgl) {
        uint64_t paddr = sg_dma_address(tmp2_sgl) + offset;
        dbg_info("VADDR: 0x%llx, DMA_ADDR: 0x%llx, size: 0x%llx, offset: %u\n", temp_vaddr2, paddr, sg_dma_len(tmp2_sgl), tmp2_sgl->offset);
        temp_vaddr2 += sg_dma_len(tmp2_sgl);
        tmp2_sgl = sg_next(tmp2_sgl);
    }

    dbg_info("coalescing");
    do {
        if(sg_dma_address(sgl) == 0xffffffffffffffff) { //manage a specific AMDGPU driver bug that issues an empty TLB entry with a DMA address of all 1s.
            sgl = sg_next(sgl);
            continue;
        }
        unsigned int tmp = sg_dma_len(sgl);
        int num_entries = 1;
        uint64_t dma_addr = sg_dma_address(sgl) + offset;
        if(tmp < 2*1024*1024) {
            tmp_sgl = sg_next(sgl);
            while (tmp < 2 *1024 * 1024) {
                if(sg_dma_address(tmp_sgl) + offset == dma_addr + tmp) {
                    tmp += sg_dma_len(tmp_sgl);
                    num_entries++;
                    tmp_sgl = sg_next(tmp_sgl);
                } else {
                    break;
                }
            }
        }

        dbg_info("contiguous region found: dma_addr: 0x%lx, length: 0x%lx, num_entries: %d\n", dma_addr, tmp, num_entries);
        
        //2MB entries
        while(tmp / (2 * 1024 * 1024) > 0) {
            tlb_set_entry(d, temp_vaddr, dma_addr, cpid, pid, true);
            tmp -= 2 * 1024 * 1024;
            dma_addr += 2 * 1024 * 1024;
            temp_vaddr += 2 * 1024 * 1024;
        }

        //4kB entries
        while(tmp / (4 * 1024) > 0) {
            tlb_set_entry(d, temp_vaddr, dma_addr, cpid, pid, false);
            tmp -= 4 * 1024;
            dma_addr += 4 * 1024;
            temp_vaddr += 4 * 1024;
        }

        //ignore coalesced entries
        for(j = 0; j < num_entries; j++) {
            sgl = sg_next(sgl);
        }
    } while(sgl);

    dbg_info("terminated");
    return rc;
}

/**
 * @brief Detach from a given DMABuf
 * 
 * @param d - the vFPGA
 * @param buf_fd - the file descriptor of the DMABuf, given by the exporter
 * @param cpid - Coyote PID
 * @param dirtied - modified TLB entries. TODO: manage dirtied entries
 */ 
int p2p_detach_dma_buf(struct fpga_dev *d, uint64_t buf_fd, int32_t cpid, int dirtied) {
    int rc = 0;
    
    dma_buf_context * context = get_dma_buf_context(gpu_buffers_dma_buf_list, buf_fd);
    if(context == NULL) {
        pr_err("Error: invalid DMABuf file descriptor.");
        return -EINVAL;
    }
    

    int i = 0;
    void * temp_vaddr = context->vaddr;

    //get coalesced entries
    int j, k;
    void * temp_vaddr2 = temp_vaddr;
    struct scatterlist * tmp_sgl, *tmp2_sgl;
    struct scatterlist * sgl = context->sgt->sgl;
    do {
        if(sg_dma_address(sgl) == 0xffffffffffffffff) { //amdgpu driver bug
            sgl = sg_next(sgl);
            continue;
        }
        unsigned int tmp = sg_dma_len(sgl);
        int num_entries = 1;
        uint64_t dma_addr = sg_dma_address(sgl) + context->offset;
        if(tmp < 2*1024*1024) {
            tmp_sgl = sg_next(sgl);
            while (tmp < 2 *1024 * 1024) {
                if(sg_dma_address(tmp_sgl) + context->offset == dma_addr + tmp) {
                    tmp += sg_dma_len(tmp_sgl);
                    num_entries++;
                    tmp_sgl = sg_next(tmp_sgl);
                } else {
                    break;
                }
            }
        }

        //2MB entries to delete
        while(tmp / (2 * 1024 * 1024) > 0) {
            tlb_delete_entry(d, temp_vaddr, cpid, dirtied, true);
            tmp -= 2 * 1024 * 1024;
            dma_addr += 2 * 1024 * 1024;
            temp_vaddr += 2 * 1024 * 1024;
        }

        //4kB entries to delete
        while(tmp / (4 * 1024) > 0) {
            tlb_delete_entry(d, temp_vaddr, cpid, dirtied, false);
            tmp -= 4 * 1024;
            dma_addr += 4 * 1024;
            temp_vaddr += 4 * 1024;
        }

        //ignore coalesced entries
        for(j = 0; j < num_entries; j++) {
            sgl = sg_next(sgl);
        }
    } while(sgl);

    //unmap buffer from FPGA bus address space
    dma_resv_lock(context->buf->resv, NULL);
    dma_buf_unmap_attachment(context->dma_attach, context->sgt, DMA_BIDIRECTIONAL);
    dma_resv_unlock(context->buf->resv);
    
    //detach FPGA from DMABuf
    kfree(context->dma_attach->importer_priv);
    dma_buf_detach(context->buf, context->dma_attach);
    
    //decrease DMABuf refcount
    dma_buf_put(context->buf);
    
    //delete context from the linked list
    delete_dma_buf_context(&gpu_buffers_dma_buf_list, buf_fd);
    
    return 0;
}


/** 
 * @brief Set TLB entry for 1 given memory page
 * 
 * @param d - vFPGA
 * @param start_vaddr - starting vaddr
 * @param start_paddr - starting paddr
 * @param cpid - Coyote PID
 * @param pid - user PID
 */
int tlb_set_entry(struct fpga_dev *d, uint64_t start_vaddr, uint64_t start_paddr, int32_t cpid, pid_t pid, bool huge)
{
    int ret_val = 0;
    int n_pages;
    struct user_pages *user_pg;
    struct task_struct *curr_task;
    uint64_t *map_array;
    struct bus_drvdata *pd;

    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);

    // context
    curr_task = pid_task(find_vpid(pid), PIDTYPE_PID);
    //dbg_info("pid found = %d", pid);

    n_pages = 1; //only 1 page to map

    // alloc
    user_pg = kzalloc(sizeof(struct user_pages), GFP_KERNEL);
    BUG_ON(!user_pg);

    user_pg->hpages = NULL; 
    user_pg->vaddr = start_vaddr;
    user_pg->n_pages = n_pages;
    user_pg->huge = huge;
    user_pg->cpid = cpid;
    // card alloc
    if(pd->en_mem) {
        user_pg->cpages = kzalloc(n_pages * sizeof(uint64_t), GFP_KERNEL);
        if (user_pg->cpages == NULL) {
            dbg_info("card buffer %d could not be allocated\n", 0);
            return -ENOMEM;
        }
        if(huge) {
            ret_val = card_alloc(d, user_pg->cpages, n_pages, LARGE_CHUNK_ALLOC);
        } else {
            ret_val = card_alloc(d, user_pg->cpages, n_pages, SMALL_CHUNK_ALLOC);
        }
        
        if (ret_val) {
            dbg_info("could not get all card pages, %d\n", ret_val);
            goto fail_card_unmap;
        }
        dbg_info("card allocated %d regular pages\n", n_pages);
    }

    // map array
    map_array = (uint64_t *)kzalloc(n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
    if (map_array == NULL) {
        dbg_info("map buffers could not be allocated\n");
        return -ENOMEM;
    }
    //dbg_info("Creating entry: vaddr %x, paddr %x\n", start_vaddr, start_paddr);
    // fill mappings
    if(huge) {
        dbg_info("creating huge entry");
        tlb_create_map(pd->ltlb_order, start_vaddr, start_paddr, (pd->en_mem ? user_pg->cpages[0] : 0), cpid, &map_array[2*0]);
    
        // fire
        tlb_service_dev(d, pd->ltlb_order, map_array, n_pages);
    } else {
        dbg_info("creating 4kB entry");
        tlb_create_map(pd->stlb_order, start_vaddr, start_paddr, (pd->en_mem ? user_pg->cpages[0] : 0), cpid, &map_array[2*0]);
        
        // fire
        tlb_service_dev(d, pd->stlb_order, map_array, n_pages);
    }

    // free
    kfree((void *)map_array);
    

    hash_add(user_sbuff_map[d->id], &user_pg->entry, start_vaddr);

    return n_pages;

fail_card_unmap:
kfree(user_pg->cpages);

    return -ENOMEM;

}

/** 
 * @brief Get user pages and fill TLB
 * 
 * @param d - vFPGA
 * @param start - starting vaddr
 * @param count - number of pages to map
 * @param cpid - Coyote PID
 * @param pid - user PID
 */
int tlb_get_user_pages(struct fpga_dev *d, uint64_t start, size_t count, int32_t cpid, pid_t pid)
{
    int ret_val = 0, i, j;
    int n_pages, n_pages_huge;
    uint64_t first;
    uint64_t last;
    struct user_pages *user_pg;
    struct vm_area_struct *vma_area_init;
    int hugepages;
    uint64_t *hpages_phys;
    uint64_t curr_vaddr, last_vaddr;
    struct task_struct *curr_task;
    struct mm_struct *curr_mm;
    uint64_t *map_array;
    uint64_t vaddr_tmp;
    struct bus_drvdata *pd;


    BUG_ON(!d);
    pd = d->pd;
    BUG_ON(!pd);
    struct device *dev = &d->pd->pci_dev->dev;
    // context
    curr_task = pid_task(find_vpid(pid), PIDTYPE_PID);
    dbg_info("pid found = %d", pid);
    curr_mm = curr_task->mm;

    // hugepages?
    vma_area_init = find_vma(curr_mm, start);
    hugepages = is_vm_hugetlb_page(vma_area_init);

    // number of pages
    first = (start & PAGE_MASK) >> PAGE_SHIFT;
    last = ((start + count - 1) & PAGE_MASK) >> PAGE_SHIFT;
    n_pages = last - first + 1;

    if(hugepages) {
        if(n_pages > MAX_N_MAP_HUGE_PAGES)
            n_pages = MAX_N_MAP_HUGE_PAGES;
    } else {
        if(n_pages > MAX_N_MAP_PAGES)
            n_pages = MAX_N_MAP_PAGES;
    }

    if (start + count < start)
        return -EINVAL;
    if (count == 0)
        return 0;

    // alloc
    user_pg = kzalloc(sizeof(struct user_pages), GFP_KERNEL);
    BUG_ON(!user_pg);

    user_pg->hpages = kcalloc(n_pages, sizeof(*user_pg->hpages), GFP_KERNEL);
    if (user_pg->hpages == NULL) {
        return -1;
    }
    //dbg_info("allocated %lu bytes for page pointer array for %d pages @0x%p, passed size %ld.\n",
             //         n_pages * sizeof(*user_pg->hpages), n_pages, user_pg->hpages, count);

    //dbg_info("pages=0x%p\n", user_pg->hpages);
    //dbg_info("first = %llx, last = %llx\n", first, last);

    for (i = 0; i < n_pages - 1; i++) {
        user_pg->hpages[i] = NULL;
    }

    // pin
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,9,0)
    ret_val = get_user_pages_fast( (unsigned long)start, n_pages, 1, user_pg->hpages); //replaces get_user_pages_remote, which is deprecated
#else 
    ret_val = get_user_pages_fast( (unsigned long)start, n_pages, 1, user_pg->hpages);
#endif
    //ret_val = get_user_pages_fast(curr_mm, (unsigned long)start, n_pages, 1, user_pg->hpages);
    //ret_val = pin_user_pages_remote(curr_mm, (unsigned long)start, n_pages, 1, user_pg->hpages);
    dbg_info("get_user_pages_fast(%llx, n_pages = %d, page start = %lx, hugepages = %d)\n", start, n_pages, page_to_pfn(user_pg->hpages[0]), hugepages);



    if(ret_val < n_pages) {
        dbg_info("could not get all user pages, %d\n", ret_val);
        goto fail_host_unmap;
    }

    // flush cache
    for(i = 0; i < n_pages; i++)
        flush_dcache_page(user_pg->hpages[i]);

    // add mapped entry
    user_pg->vaddr = start;
    user_pg->n_hpages = n_pages;
    user_pg->huge = hugepages;

    vaddr_tmp = start;

    // huge pages
    if (hugepages) {
        first = (start & pd->ltlb_order->page_mask) >> pd->ltlb_order->page_shift;
        last = ((start + count - 1) & pd->ltlb_order->page_mask) >> pd->ltlb_order->page_shift;
        n_pages_huge = last - first + 1;
        user_pg->n_pages = n_pages_huge;

        // prep hpages
        hpages_phys = kzalloc(n_pages_huge * sizeof(uint64_t), GFP_KERNEL);
        if (hpages_phys == NULL) {
            dbg_info("card buffer %d could not be allocated\n", i);
            return -ENOMEM;
        }

        j = 0;
        curr_vaddr = start;
        last_vaddr = -1;
        for (i = 0; i < n_pages; i++) {
            if (((curr_vaddr & pd->ltlb_order->page_mask) >> pd->ltlb_order->page_shift) != ((last_vaddr & pd->ltlb_order->page_mask) >> pd->ltlb_order->page_shift)) {
                hpages_phys[j] = dma_map_page(dev, user_pg->hpages[i], 0, 2*1024*1024, DMA_BIDIRECTIONAL); //For HW IOMMU setup.
                dbg_info("hugepage %d at %llx\n", j, hpages_phys[j]);
                last_vaddr = curr_vaddr;
                j++;
            }
            curr_vaddr += PAGE_SIZE;
        }


        // card alloc
        if(pd->en_mem) {
            user_pg->cpages = kzalloc(n_pages_huge * sizeof(uint64_t), GFP_KERNEL);
            if (user_pg->cpages == NULL) {
                dbg_info("card buffer %d could not be allocated\n", i);
                return -ENOMEM;
            }

            ret_val = card_alloc(d, user_pg->cpages, n_pages_huge, LARGE_CHUNK_ALLOC);
            if (ret_val) {
                dbg_info("could not get all card pages, %d\n", ret_val);
                goto fail_card_unmap;
            }
            dbg_info("card allocated %d hugepages\n", n_pages_huge);
        }

        // map array
        map_array = (uint64_t *)kzalloc(n_pages_huge * 2 * sizeof(uint64_t), GFP_KERNEL);
        if (map_array == NULL) {
            dbg_info("map buffers could not be allocated\n");
            return -ENOMEM;
        }

        // fill mappings
        for (i = 0; i < n_pages_huge; i++) {
            if(!hpages_phys[i]) {
                dbg_info("ERROR! null ptr\n");
                continue;
            }
            tlb_create_map(pd->ltlb_order, vaddr_tmp, hpages_phys[i], (pd->en_mem ? user_pg->cpages[i] : 0), cpid, &map_array[2*i]);
            vaddr_tmp += pd->ltlb_order->page_size;
        }

        // fire
        tlb_service_dev(d, pd->ltlb_order, map_array, n_pages_huge);

        // free
        kfree((void *)map_array);
    
    // small pages
    } else {
        user_pg->n_pages = n_pages;

        // card alloc
        if(pd->en_mem) {
            user_pg->cpages = kzalloc(n_pages * sizeof(uint64_t), GFP_KERNEL);
            if (user_pg->cpages == NULL) {
                dbg_info("card buffer %d could not be allocated\n", i);
                return -ENOMEM;
            }

            ret_val = card_alloc(d, user_pg->cpages, n_pages, SMALL_CHUNK_ALLOC);
            if (ret_val) {
                dbg_info("could not get all card pages, %d\n", ret_val);
                goto fail_card_unmap;
            }
            dbg_info("card allocated %d regular pages\n", n_pages);
        }

        // map array
        map_array = (uint64_t *)kzalloc(n_pages * 2 * sizeof(uint64_t), GFP_KERNEL);
        if (map_array == NULL) {
            dbg_info("map buffers could not be allocated\n");
            return -ENOMEM;
        }

        // fill mappings
        for (i = 0; i < n_pages; i++) {
            void * dma_addr = (void *) dma_map_page(dev, user_pg->hpages[i], 0, 4*1024, DMA_BIDIRECTIONAL); //For HW IOMMU setup.
            tlb_create_map(pd->stlb_order, vaddr_tmp, dma_addr, (pd->en_mem ? user_pg->cpages[i] : 0), cpid, &map_array[2*i]);
            vaddr_tmp += PAGE_SIZE;
        }

        // fire
        tlb_service_dev(d, pd->stlb_order, map_array, n_pages);

        // free
        kfree((void *)map_array);
    }

    hash_add(user_sbuff_map[d->id], &user_pg->entry, start);

    return n_pages;

fail_host_unmap:
    // release host pages
    for(i = 0; i < ret_val; i++) {
        put_page(user_pg->hpages[i]);
    }

    kfree(user_pg->hpages);

    return -ENOMEM;

fail_card_unmap:
    // release host pages
    for(i = 0; i < user_pg->n_hpages; i++) {
        put_page(user_pg->hpages[i]);
    }

    kfree(user_pg->hpages);
    kfree(user_pg->cpages);

    return -ENOMEM;
}

/**
 * @brief DMABuf exporter callback for dma_buf_dynamic_attach()
 * 
 * @param dmabuf - the exported dmabuf
 * @param attachment - the corresponding dma_buf_attachment
 * 
 */
int dma_buf_exporter_attach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment) {
	dbg_info("executed\n");
	return 0;
}

/**
 * @brief DMABuf exporter callback for dma_buf_detach()
 * 
 * @param dmabuf - the exported dmabuf
 * @param attachment - the corresponding dma_buf_attachment
 */
void dma_buf_exporter_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment) {
	dbg_info("executed\n");
	return;
}

/**
 * @brief DMABuf exporter callback used when all DMABuf importers close their sessions.
 * 
 * @param dmabuf - the exported dmabuf
 */
void dma_buf_exporter_release(struct dma_buf *dma_buf) {
    dbg_info("executed\n");

    //release internal memory
    if(dma_buf->priv != NULL) {
        kfree(dma_buf->priv);
        dma_buf->priv = NULL;
    }
    return;
}

//internal information regarding the DMABuf to export
struct dma_buf_exporter_data {
    void * vaddr; //virtual address of the CTRL registers memory area
    uint32_t size; //size of the area: see coyote_dev.h/FPGA_CTRL_SIZE
};

/**
 * @brief DMABuf exporter callback for dma_buf_map_attachment()
 * 
 * @param attachment - the dma_buf_attachment for the exported dmabuf
 * @param dim - the data direction for DMA: see https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt for "DMA Direction"
 */
struct sg_table *dma_buf_exporter_map(struct dma_buf_attachment *attachment, enum dma_data_direction dir) {
    struct sg_table *table;
    struct scatterlist *sgl;
    struct dma_buf_exporter_data * data;
    struct page * pages;
    int i, ret_val;

    //get internal data
    data = attachment->dmabuf->priv;
    if(!data) {
        pr_err("pointer to data is null\n");
        return -ENOMEM;
    }

    //allocate and build scattergather table

    dbg_info("allocating sg_table\n");

    table = kmalloc(sizeof(struct sg_table), GFP_KERNEL);
    if(!table) {
        pr_err("cannot allocate table\n");
        return -ENOMEM;
    }

    int num_pages = PAGE_ALIGN(data->size) / PAGE_SIZE;

    dbg_info("num_pages for CTRL region: %d\n", num_pages);

    if(sg_alloc_table(table, num_pages, GFP_KERNEL)) {
        kfree(table);
        pr_err("cannot allocate table, after kmalloc\n");
        return -ENOMEM;
    }

    sgl = table->sgl;

    dbg_info("building table\n");

//This should be useless for our purpose
//     #if LINUX_VERSION_CODE >= KERNEL_VERSION(5,9,0)
//     ret_val = get_user_pages_fast((unsigned long)data->vaddr, num_pages, 1, &pages);
// #else 
//     ret_val = get_user_pages_fast((unsigned long)data->vaddr, num_pages, 1, &pages);
// #endif

//     dbg_info("dma_buf_exporter_map(): retval = %d", ret_val);


    pgd_t * pgd;
    pmd_t * pmd;
    pte_t * pte;
    p4d_t * p4d;
    pud_t * pud;

    struct mm_struct *mm = current->mm;

    //find struct page * for CTRL memory area, as it can be done by using internal Linux data structure
    for(i = 0; i < num_pages; i++) {
        pgd = pgd_offset(mm, data->vaddr + i * PAGE_SIZE);
        p4d = p4d_offset(pgd, data->vaddr + i * PAGE_SIZE);
        pud = pud_offset(p4d, data->vaddr + i * PAGE_SIZE);
        pmd = pmd_offset(pud, data->vaddr + i * PAGE_SIZE);  
        pte = pte_offset_map(pmd, data->vaddr + i * PAGE_SIZE); 
        struct page * pag = pte_page(*pte); 

        sg_set_page(sgl, pag, PAGE_SIZE, 0);
        
        dbg_info("vaddr: %lx is valid ? %d\n",data->vaddr + i * PAGE_SIZE, virt_addr_valid(data->vaddr + i * PAGE_SIZE) ); //Linux says this register is not a valid virtual address, I am not sure if this is an issue
        
        dma_addr_t addr = dma_map_page(attachment->dev, pag, 0, PAGE_SIZE, DMA_BIDIRECTIONAL); //map CTRL register area into GPU memory
        sgl->dma_address = addr;
        sgl->dma_length = PAGE_SIZE;
        sgl->length = PAGE_SIZE;
        sgl->offset = (unsigned int)((unsigned long) (data->vaddr + i * PAGE_SIZE) & (unsigned int) ~PAGE_MASK);
        dbg_info("dma_address = %lx, dma_length = %lx, dma_offset = %lx\n", sgl->dma_address, sgl->dma_length, sgl->offset);
        sg_dma_mark_bus_address(sgl);
        sgl = sg_next(sgl);
    }

    dbg_info("terminated\n");

    return table;
}

/**
 * @brief DMABuf exporter callback for dma_buf_unmap_attachment()
 * 
 * @param attachment - the dma_buf_attachment for the exported dmabuf
 * @param table - the scattergather table of the mapping
 * @param dim - the data direction for DMA: see https://www.kernel.org/doc/Documentation/DMA-API-HOWTO.txt for "DMA Direction"
 */
void dma_buf_exporter_unmap(struct dma_buf_attachment *attachment, struct sg_table *table, enum dma_data_direction dir) {
    dbg_info("unmapping dma_buf\n");
    dma_unmap_sg(attachment->dev, table->sgl, table->nents, dir);
    sg_free_table(table);
    kfree(table);
    dbg_info("terminated\n");
    return;
}

// Data structure required to associated the DMABuf exporter with its callbacks
const struct dma_buf_ops exporter_ops = {
    .attach = dma_buf_exporter_attach,
	.detach = dma_buf_exporter_detach,
	.map_dma_buf = dma_buf_exporter_map,
	.unmap_dma_buf = dma_buf_exporter_unmap,
	.release = dma_buf_exporter_release
};

/**
 * Export a DMABuf related to internal FPGA registers for DMA
 * 
 * @param d - the vFPGA
 * @param vaddr - the virtual address of the FPGA registers memory area: e.g., CTRL, CNFG, etc.
 * @param size - the size of the memory area
 */
unsigned long dma_buf_export_regs(struct fpga_dev *d, void * vaddr, uint32_t size) {

    struct dma_buf * buf;
    struct dma_buf_exporter_data * data;

    dbg_info("allocating dma_buf data\n");

    data = kmalloc(sizeof(struct dma_buf_exporter_data), GFP_KERNEL);
    if(!data) {
        dbg_info("allocation of data failed\n");
        return -ENOMEM;
    }

    data->vaddr = vaddr;
    data->size = size;

    DEFINE_DMA_BUF_EXPORT_INFO(export_info);

    export_info.owner = THIS_MODULE;
    export_info.ops = &exporter_ops;
    export_info.size = size;
    export_info.flags = O_CLOEXEC;
    export_info.resv = NULL;
    export_info.priv = data;

    //export DMABuf
    dbg_info("exporting dma_buf\n");
    buf = dma_buf_export(&export_info);

    if (IS_ERR(buf)) {
		pr_err("failed to export dma_buf\n");
		goto err;
	}

    //open DMABuf and retrieve its file descriptor
    unsigned long fd = dma_buf_fd(buf, O_CLOEXEC);

    dbg_info("terminated\n");

    return fd;

err:

    kfree(data);

    return -ENOMEM;
}

/**
 * Close the exported for DMABuf related to the FPGA internal registers memory area
 * TODO: eliminate this IOCTL call as it seems not to be required
 * 
 * @param dmabuf_fd - the file descriptor for the exported DMABuf
 */
int dma_buf_export_close(uint32_t dma_buf_fd) {
        
    dbg_info("dma_buf_export_close() terminated\n");

    return 0;
}

// //DEBUG: Linked list to keep track of the imported DMABufs from the FPGA Coyote driver
// dma_buf_context * fpga_memory_dma_buf_list = NULL;

// //DEBUG
// /**
//  * @brief Imported the DMABuf related to the CTRL register memory area and attach the GPU MI100 to it
//  * Unstable API
//  * 
//  * @param d - the vFPGA
//  * @param fd - the file descriptor of the exported DMABuf
//  * 
//  */
// uint64_t dma_buf_from_fd_to_gpu(struct fpga_dev *d, int fd) {
//     dbg_info("getting GPU device\n");
//     struct pci_dev * gpu_pci_dev = pci_get_device(0x1002, 0x738c, NULL); //AMD Instinct MI100
    
//     struct device * gpu_dev = NULL; 
//     if(!gpu_pci_dev) {
//         pr_err("getting GPU device failed!\n");
//         return -1;
//     }
//     gpu_dev = &gpu_pci_dev->dev;
//     //struct device * fpga_dev = &d->pd->pci_dev->dev;
//     int rc = 0;
   
//     int err = 0;

//     int dma_buf_fd = fd;

//     dbg_info("fd = %d\n", fd);

//     struct dma_buf * buf = dma_buf_get(dma_buf_fd);
//     if(buf == ERR_PTR || buf == NULL) {
//         pr_err("ERROR: dma_buf_get failed.\n");
//         return -EINVAL;
//     }
//     dbg_info("dma_buf_get executed.\nSize:%llx,name:%s\n", buf->size, buf->name);
//         struct dma_buf_attachment *dma_attach;
//     // dbg_info(KERN_INFO "dma_buf_from_fd_to_gpu(): GPU device address: %llx\n",  gpu_dev);
//     // dbg_info(KERN_INFO "dma_buf_from_fd_to_gpu(): GPU name: %s, parent's name: %s, list head next: %llx\n ", gpu_dev->init_name, gpu_dev->parent->init_name, gpu_dev->devres_head.next);
//     //dma_attach = dma_buf_attach(buf, fpga_dev);
//     dma_attach = dma_buf_attach(buf, &gpu_pci_dev->dev);
//     if(dma_attach == ERR_PTR || dma_attach == NULL) {
//         pr_err("ERROR: dma_buf_attach failed.\n");
//         dma_buf_put(buf);
//         return -EINVAL;
//     }
    
//     dbg_info("dma_buf_attach executed.\n");
//     struct sg_table *sgt = dma_buf_map_attachment(dma_attach, DMA_BIDIRECTIONAL);
//     if(sgt == NULL) {
//         pr_err("dma_buf_from_fd_to_gpu(): ERROR: sg_table is NULL.\n");
//         dma_buf_detach(buf, dma_attach);
//         dma_buf_put(buf);
//         return -EINVAL;
//     }
//     dbg_info("dma_buf_map_attachment executed.\n");
//     struct scatterlist *sgl = sgt->sgl;
//     struct scatterlist * tmp_sgl;
//     tmp_sgl = sgl;

//     while(tmp_sgl) {
//         uint64_t paddr = sg_dma_address(tmp_sgl);
//         dbg_info("DMA_ADDR: 0x%llx, size: 0x%llx, offset: %u\n", paddr, sg_dma_len(tmp_sgl), tmp_sgl->offset);
//         tmp_sgl = sg_next(tmp_sgl);
//     }

//     dbg_info("creating dma_buf context.\n");

//     if(create_dma_buf_context(&fpga_memory_dma_buf_list, dma_buf_fd, buf, dma_attach, sgt, 0x0, 0) == NULL) {
//         pr_err("Error in dma_buf_context creation!");
//         return -ENOMEM;
//     }

//     uint64_t final_addr = (uint64_t) sg_dma_address(sgl);

//     dbg_info("returning %lx\n", final_addr);

//     return final_addr;

// }

// //DEBUG
// /**
//  * @brief Detach the GPU from the DMABuf related to the CTRL register memory area
//  * Unstable API
//  * 
//  * @param d - the vFPGA
//  * @param buf_fd - the file descriptor of the exported DMABuf
//  * @param cpid - the Coyote PID
//  * 
//  */
// int gpu_detach_dma_buf(struct fpga_dev *d, uint64_t buf_fd, int32_t cpid) {
//     int rc = 0;
//     dbg_info("gpu_detach_dma_buf()");

//     dma_buf_context * context = get_dma_buf_context(fpga_memory_dma_buf_list, buf_fd);
//     if(context == NULL) {
//         pr_err("Error! Invalid DMABuf fd.");
//         return -EINVAL;
//     }
//     dbg_info("dma_buf: %llx, dma_attach: %llx, sgt: %llx, vaddr: %llx\n", context->buf, context->dma_attach, context->sgt, context->vaddr);
    

//     int i = 0;
//     void * temp_vaddr = context->vaddr;

//     int j, k;
//     void * temp_vaddr2 = temp_vaddr;
    
//     dma_buf_unmap_attachment(context->dma_attach, context->sgt, DMA_BIDIRECTIONAL);
//     dbg_info("dma_buf_unmap_attachment executed");
//     dma_buf_detach(context->buf, context->dma_attach);
//     dbg_info("dma_buf_detach executed");
//     dma_buf_put(context->buf);
//     dbg_info("dma_buf_put executed");
//     delete_dma_buf_context(&fpga_memory_dma_buf_list, buf_fd);
//     dbg_info("terminated");
//     return 0;
// }
