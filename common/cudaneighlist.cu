#include "cudaneighlist.h"
#include "thrust/scan.h"
#include "hostneighlist.h"

#include <cassert>
#include <cmath>

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
  int *valid
) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x);

  if (tid < nparticles) {
    for (int i=0; i<numneigh[tid]; i++) {
      valid[offset[tid]+i] = 1;
    }
  }
}

CudaNeighList::CudaNeighList(
  int block_size,
  int nparticles, int maxpage, int pgsize) :
  nparticles(nparticles), maxpage(maxpage), pgsize(pgsize),
  block_size(block_size),
  grid_size(min(nparticles/block_size, MAX_GRID_DIM),
            max((int)ceil(((float)nparticles/block_size)/MAX_GRID_DIM), 1)),

  d_numneigh_size(nparticles * sizeof(int)),
  d_firstneigh_size(nparticles * sizeof(int *)),
  d_pages_size(maxpage * sizeof(int *)),
  d_pageidx_size(d_numneigh_size),
  d_offset_size(d_numneigh_size),
  d_neighidx_size(maxpage * pgsize * sizeof(int)),
  d_valid_size(d_neighidx_size),

  tload(0), tunload(0), tdecode(0)
{
  cudaMalloc((void **)&d_numneigh, d_numneigh_size);
  cudaMalloc((void **)&d_firstneigh, d_firstneigh_size);
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_pageidx, d_pageidx_size);
  cudaMalloc((void **)&d_offset, d_offset_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
  cudaMalloc((void **)&d_valid, d_valid_size);
}

CudaNeighList::~CudaNeighList() {
  cudaFree(d_numneigh);
  cudaFree(d_firstneigh);
  cudaFree(d_pages);
  cudaFree(d_pageidx);
  cudaFree(d_offset);
  cudaFree(d_neighidx);
  cudaFree(d_valid);
}

/*
 * Make bigger device arrays
 * NB: does not update maxpage!
 */
void CudaNeighList::resize(int new_maxpage) {
  cudaFree(d_pages);
  cudaFree(d_neighidx);
  cudaFree(d_valid);
  d_pages_size = new_maxpage * sizeof(int *);
  d_neighidx_size = new_maxpage * pgsize * sizeof(int);
  d_valid_size    = d_neighidx_size;
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
  cudaMalloc((void **)&d_valid, d_valid_size);
}

/*
 *    CPU            DEV
 *                                                                            ,-------. 
 *                                                                            v       |
 *    numneigh ----> [d_numneigh] -----------------------------(scan) --> [d_offset]  |
 *                                                               /            |       |
 *  .(if maxpage > 1).........................................  /           (dec2) ---'
 *  | firstneigh -->  d_firstneigh -- (dec1) --> [d_pageidx] --'            (dec3)
 *  `........                           /                    |
 *          |                          /                     |
 *    pages |------>  d_pages --------'                      |
        \   `................................................'
 *        \
 *         '-------> [d_neighidx]
 *
 * dec1 = decode_neighlist_p1
 * scan = maxpage > 1 ? exclusive_scan : exclusive_scan_by_key
 * dec2 = decode_neighlist_p2
 * dec3 = decode_neighlist_p3
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
  // simulate third part of decode
  int *h_valid = new int[maxpage*pgsize];
  std::fill_n(h_valid, maxpage*pgsize, 0);
  for (int i=0; i<nparticles; i++) {
    for (int k=0; k<numneigh[i]; k++) {
      h_valid[h_offset[i]+k] = 1;
    }
  }
  cudaMemcpy(d_valid, h_valid, d_valid_size, cudaMemcpyHostToDevice);
  delete[] h_offset;
  delete[] h_valid;
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
  cudaMemset(d_valid, 0, d_valid_size);
  decode_neighlist_p3<<<grid_size, block_size>>>(
    nparticles,
    d_numneigh,
    d_offset,
    d_valid);
#endif

#if PARANOID
  check_decode(numneigh, firstneigh);
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
