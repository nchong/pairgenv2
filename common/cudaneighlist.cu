#include "cudaneighlist.h"
#include "thrust/scan.h"
#include "hostneighlist.h"

#include <cassert>
#include <cmath>
#include <cstdio>

#ifndef MAX_GRID_DIM
#error You need to #define MAX_GRID_DIM (see Makefile.config)
#endif

__global__ void decode_neighlist_p1(
  //inputs
  int nparticles,
  int **firstneigh, //nb: contains cpu pointers: do not dereference!
  int maxpage,
  int **pages,      //nb: contains cpu pointers: do not dereference!
  int pgsize,
  //outputs
  int *pageidx
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  if (tid < nparticles) {
    int *myfirstneigh = firstneigh[tid];
    int mypage = -1;
    for (int p=0; p<maxpage; p++) {
      if ( (pages[p] <= myfirstneigh) &&
                       (myfirstneigh < (pages[p]+pgsize)) ) {
        mypage = p;
      }
    }
    pageidx[tid] = mypage;
  }
}

__global__ void decode_neighlist_p2(
  //inputs
  int nparticles,
  int *pageidx,
  int pgsize,
  //inout
  int *offset
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  if (tid < nparticles) {
    int mypage = pageidx[tid];
    offset[tid] += (mypage*pgsize);
  }
}

__global__ void decode_neighlist_p3(
  //inputs
  int nparticles,
  int *numneigh,
  int *offset,
  //inout
  int *valid,
  int *dati
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  if (tid < nparticles) {
    for (int i=0; i<numneigh[tid]; i++) {
      valid[offset[tid]+i] = 1;
      dati[offset[tid]+i] = tid;
    }
  }
}

#if TPN
__global__ void invert_neighlist_p1(
  //inputs
  int nparticles,
  int *neighidx, int *offset, int *numneigh,
  //outputs
  int *nel       //nb: must be zeroed!
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  int n = numneigh[tid];
  if (tid < nparticles && n > 0) {
    int off = offset[tid];
    for (int k=0; k<n; k++) {
      int j = neighidx[off+k];
      atomicAdd(&nel[j], 1);
    }
  }
}

__global__ void invert_neighlist_p2(
  //inputs
  int nneighbors,
  int *valid,
  int *neighidx,
  int *ffo,
  //outputs
  int *tad,
  int *nel       //nb: must be zeroed!
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  if (tid < nneighbors && valid[tid]) {
    int j = neighidx[tid];
    int k = atomicAdd(&nel[j], 1);
    tad[ffo[j]+k] = tid;
  }
}

__global__ void invert_neighlist_p2_tpa(
  //inputs
  int nparticles,
  int *datj,
  int *off,
  int *len,
  int *ffo,
  //outputs
  int *tad,
  int *nel       //nb: must be zeroed!
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  int n = len[tid];
  if (tid < nparticles && n > 0) {
    int off = offset[tid];
    for (int i=0; i<n; i++) {
      int j = datj[off+i];
      int k = atomicAdd(&nel[j], 1);
      tad[ffo[j]+k] = tid;
    }
  }
}
#endif

CudaNeighList::CudaNeighList(
  int block_size,
  int nparticles, int maxpage, int pgsize) :
  nparticles(nparticles), maxpage(maxpage), pgsize(pgsize),
  block_size(block_size),
  grid_size(min(nparticles/block_size, MAX_GRID_DIM),
            max((int)ceil(((float)nparticles/block_size)/MAX_GRID_DIM), 1)),
#if TPN
  neighbor_grid_size(
            min((maxpage*pgsize)/block_size, MAX_GRID_DIM),
            max((int)ceil(((float)(maxpage*pgsize)/block_size)/MAX_GRID_DIM), 1)),
#endif

  d_numneigh_size(nparticles * sizeof(int)),
  d_firstneigh_size(nparticles * sizeof(int *)),
  d_pages_size(maxpage * sizeof(int *)),
  d_pageidx_size(d_numneigh_size),
  d_offset_size(d_numneigh_size),
  d_neighidx_size(maxpage * pgsize * sizeof(int)),
#if TPN
  d_valid_size(d_neighidx_size),
  d_dati_size(d_neighidx_size),
  d_tad_size(d_neighidx_size),
  d_ffo_size(d_numneigh_size),
  d_nel_size(d_numneigh_size),
#endif

  tload(0), tunload(0), tdecode(0)
{
  cudaMalloc((void **)&d_numneigh, d_numneigh_size);
  cudaMalloc((void **)&d_firstneigh, d_firstneigh_size);
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_pageidx, d_pageidx_size);
  cudaMalloc((void **)&d_offset, d_offset_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
#if TPN
  cudaMalloc((void **)&d_valid, d_valid_size);
  cudaMalloc((void **)&d_dati, d_dati_size);
  cudaMalloc((void **)&d_tad, d_tad_size);
  cudaMalloc((void **)&d_ffo, d_ffo_size);
  cudaMalloc((void **)&d_nel, d_nel_size);
#endif
}

CudaNeighList::~CudaNeighList() {
  cudaFree(d_numneigh);
  cudaFree(d_firstneigh);
  cudaFree(d_pages);
  cudaFree(d_pageidx);
  cudaFree(d_offset);
  cudaFree(d_neighidx);
#if TPN
  cudaFree(d_valid);
  cudaFree(d_dati);
  cudaFree(d_tad);
  cudaFree(d_ffo);
  cudaFree(d_nel);
#endif
}

/*
 * Make bigger device arrays (too bad CUDA has no realloc)
 * NB: does not update maxpage!
 */
void CudaNeighList::resize(int new_maxpage) {
  cudaFree(d_pages);
  cudaFree(d_neighidx);
  d_pages_size = new_maxpage * sizeof(int *);
  d_neighidx_size = new_maxpage * pgsize * sizeof(int);
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
#if TPN
  cudaFree(d_valid);
  cudaFree(d_dati);
  cudaFree(d_tad);
  d_valid_size    = d_neighidx_size;
  d_dati_size     = d_neighidx_size;
  d_tad_size      = d_neighidx_size;
  cudaMalloc((void **)&d_valid, d_valid_size);
  cudaMalloc((void **)&d_dati, d_dati_size);
  cudaMalloc((void **)&d_tad, d_tad_size);
#endif
}

/*
 *    CPU            DEV
 *                                                                ,-------.
 *                                                                v       |
 *    numneigh ====> [d_numneigh] -----------------(scan) --> [d_offset]  |
 *                                                   ^            |       |
 *                                                   |            v       |
 *  .(if maxpage > 1).............................   |          (dec2) ---'
 *  : firstneigh ==>  d_firstneigh               :   |
 *  `........           \                        :   |
 *          :            v                       :  /
 *          :           (dec1) --> [d_pageidx] --:-'
 *          :            ^                       :
 *          :           /                        :
 *    pages =======>  d_pages                    :
        \\  `....................................:
 *       \\
 *        ''=======> [d_neighidx]
 *
 * ===> = memcpy (pages into d_neighidx is load_pages() memcpy)
 * ---> = dataflow
 * dec1 = decode_neighlist_p1 (TPA)
 * scan = maxpage > 1 ? exclusive_scan : exclusive_scan_by_key
 * dec2 = decode_neighlist_p2 (TPA)
 */
void CudaNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage) {
  // nb: we do not expect nparticles or pgsize to change
  // resize if necessary
  if (maxpage < reload_maxpage) {
    resize(reload_maxpage);
    maxpage = reload_maxpage;
  }

  cudaMemcpy(d_numneigh, numneigh, d_numneigh_size, cudaMemcpyHostToDevice);
  load_pages(d_neighidx, pages);
#if HOST_DECODE
  int *h_offset = host_decode_neighlist(nparticles, maxpage, numneigh, firstneigh, pages, pgsize);
  cudaMemcpy(d_offset, h_offset, d_offset_size, cudaMemcpyHostToDevice);
#if TPN
  // simulate third part of decode
  int *h_dati  = new int[maxpage*pgsize];
  int *h_valid = new int[maxpage*pgsize];
  std::fill_n(h_valid, maxpage*pgsize, 0);
  for (int i=0; i<nparticles; i++) {
    for (int k=0; k<numneigh[i]; k++) {
      h_dati[ h_offset[i]+k] = i;
      h_valid[h_offset[i]+k] = 1;
    }
  }
  cudaMemcpy(d_dati, h_dati, d_dati_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_valid, h_valid, d_valid_size, cudaMemcpyHostToDevice);
  delete[] h_dati;
  delete[] h_valid;
#endif
  delete[] h_offset;
#else
  thrust::device_ptr<int> thrust_numneigh(d_numneigh);
  thrust::device_ptr<int> thrust_offset(d_offset);

  if (maxpage == 1) {
    thrust::exclusive_scan(thrust_numneigh, thrust_numneigh + nparticles, thrust_offset);
  } else {
    cudaMemcpy(d_firstneigh, firstneigh, d_firstneigh_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pages, pages, d_pages_size, cudaMemcpyHostToDevice);
    decode_neighlist_p1<<<grid_size, block_size>>>(
      nparticles,
      d_firstneigh,
      maxpage,
      d_pages,
      pgsize,
      d_pageidx);
    thrust::device_ptr<int> thrust_pageidx(d_pageidx);
    thrust::exclusive_scan_by_key(
      thrust_pageidx,              // ] keys
      thrust_pageidx + nparticles, // ] 
      thrust_numneigh,             //vals
      thrust_offset);              //output
    decode_neighlist_p2<<<grid_size, block_size>>>(
      nparticles,
      d_pageidx,
      pgsize,
      d_offset);
  }
#if TPN
  cudaMemset(d_valid, 0, d_valid_size);
  decode_neighlist_p3<<<grid_size, block_size>>>(
    nparticles,
    d_numneigh,
    d_offset,
    d_valid,
    d_dati);
#endif
#endif

#if PARANOID
  check_decode(numneigh, firstneigh);
#endif

#if TPN
  reload_inverse();
  check_inverse();
#endif
}

#if TPN
/*
 * DEV
 *                 .-------------------------------.
 *                /                                v
 *               /                  .---(zero)--> (inv2) --> [d_nel]
 * [d_valid] ---'                  /                ^        [d_tad]
 *            /                   /                 |
 * [d_neighidx] --+-(inv1)--> [d_nel] --(scan)--> [d_ffo]
 *                |
 * [d_offset] ----+
 *                |
 * [d_numneigh] --+
 *
 * ---> = dataflow
 * inv1 = invert_neighlist_p1 (TPA)
 * zero = memset(0)
 * scan = exclusive_scan
 * inv2 = invert_neighlist_p2 (TPN)
 */
void CudaNeighList::reload_inverse() {
  cudaMemset(d_nel, 0, d_nel_size);
  invert_neighlist_p1<<<grid_size, block_size>>>(
    nparticles,
    d_neighidx,
    d_offset,
    d_numneigh,
    d_nel);
  thrust::device_ptr<int> thrust_nel(d_nel);
  thrust::device_ptr<int> thrust_ffo(d_ffo);
  thrust::exclusive_scan(thrust_nel, thrust_nel + nparticles, thrust_ffo);
  cudaMemset(d_nel, 0, d_nel_size);
#if 1
  invert_neighlist_p2<<<neighbor_grid_size, block_size>>>(
    (maxpage*pgsize),
    d_valid,
    d_neighidx,
    d_ffo,
    d_tad,
    d_nel);
#else
  invert_neighlist_p2_tpa<<<grid_size, block_size>>>(
    nparticles,
    d_neighidx,
    d_offset,
    d_numneigh,
    d_ffo,
    d_tad,
    d_nel);
#endif
#if PARANOID
  check_inverse();
#endif
}

/* end to end check of decode */
void CudaNeighList::check_decode(int *numneigh, int **firstneigh) {
  int *neighidx = new int[maxpage*pgsize];
  int *offset = new int[nparticles];
  cudaMemcpy(neighidx, d_neighidx, d_neighidx_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(offset, d_offset, d_offset_size, cudaMemcpyDeviceToHost);
  for (int i=0; i<nparticles; i++) {
    for (int j=0; j<numneigh[i]; j++) {
      int expected = firstneigh[i][j];
      int myoffset = offset[i];
      int actual = neighidx[myoffset+j];
      assert(expected == actual);
    }
  }
  delete[] neighidx;
  delete[] offset;
}

/* end to end check of inverse */
void CudaNeighList::check_inverse() {
  printf("check_inverse...");
  //nl
  int *valid = new int[maxpage*pgsize];
  int *datj  = new int[maxpage*pgsize];
  //inverse nl
  int *tad   = new int[maxpage*pgsize];
  int *ffo   = new int[nparticles];
  int *nel   = new int[nparticles];
  cudaMemcpy(valid, d_valid,    d_valid_size,    cudaMemcpyDeviceToHost);
  cudaMemcpy(datj,  d_neighidx, d_neighidx_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(tad,   d_tad,      d_tad_size,      cudaMemcpyDeviceToHost);
  cudaMemcpy(ffo,   d_ffo,      d_ffo_size,      cudaMemcpyDeviceToHost);
  cudaMemcpy(nel,   d_nel,      d_nel_size,      cudaMemcpyDeviceToHost);
  //check nel
  int *expected_nel = new int[maxpage*pgsize];
  for (int n=0; n<maxpage*pgsize; n++) {
    if (valid[n]) {
      expected_nel[datj[n]]++;
    }
  }
  for (int i=0; i<nparticles; i++) {
    if (expected_nel[i]  != nel[i])
        printf("%d> expected=%d actual=%d\n", i, expected_nel[i],  nel[i]);
    assert(expected_nel[i]  == nel[i]);
  }
  delete[] expected_nel;
  //end to end property
  for (int n=0; n<maxpage*pgsize; n++) {
    if (valid[n]) {
      int j = datj[n];
      bool found = false;
      for (int k=0; k<nel[j]; k++) {
        found = found || (tad[ffo[j]+k] == n);
      }
      assert(found);
    }
  }
  printf("ok\n");
  delete[] valid;
  delete[] datj;
  delete[] tad;
  delete[] ffo;
  delete[] nel;
}
#endif
