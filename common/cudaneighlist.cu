#include "cudaneighlist.h"
#include "thrust/scan.h"

#define PARANOID true
#if PARANOID
#include "scanref.h"
#include <cassert>
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
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

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

CudaNeighList::CudaNeighList(
  int block_size,
  int nparticles, int maxpage, int pgsize) :
  block_size(block_size),
  grid_size((nparticles/block_size)+1),
  nparticles(nparticles), maxpage(maxpage), pgsize(pgsize),

  d_numneigh_size(nparticles * sizeof(int)),
  d_firstneigh_size(nparticles * sizeof(int *)),
  d_pages_size(maxpage * sizeof(int *)),
  d_pagebreak_size(d_numneigh_size),
  d_pageidx_size(d_numneigh_size),
  d_offset_size(d_numneigh_size),
  d_neighidx_size(maxpage * pgsize * sizeof(int)),

  tload(0), tunload(0), tdecode(0)
{
  cudaMalloc((void **)&d_numneigh, d_numneigh_size);
  cudaMalloc((void **)&d_firstneigh, d_firstneigh_size);
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_pageidx, d_pageidx_size);
  cudaMalloc((void **)&d_offset, d_offset_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
}

CudaNeighList::~CudaNeighList() {
  cudaFree(d_numneigh);
  cudaFree(d_firstneigh);
  cudaFree(d_pages);
  cudaFree(d_pageidx);
  cudaFree(d_offset);
  cudaFree(d_neighidx);
}

/*
 * Make bigger device arrays
 * NB: does not update maxpage!
 */
void CudaNeighList::resize(int new_maxpage) {
  cudaFree(d_pages);
  cudaFree(d_neighidx);
  d_pages_size = new_maxpage * sizeof(int *);
  d_neighidx_size = new_maxpage * pgsize * sizeof(int);
  cudaMalloc((void **)&d_pages, d_pages_size);
  cudaMalloc((void **)&d_neighidx, d_neighidx_size);
}

/*
 *    CPU            DEV
 *
 *    numneigh ----> [d_numneigh] -------------------------------(scan) --> [d_offset]
 *                                                                 /
 *  .(if maxpage > 1)...........................                  /
 *  | firstneigh -->  d_firstneigh -- (decode) |-> [d_pageidx] --'
 *  `........                           /      |
 *          |                          /       |
 *    pages |------>  d_pages --------'        |
        \   |...................................
 *        \
 *         '-------> [d_neighidx]
 *
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
  thrust::device_ptr<int> thrust_numneigh(d_numneigh);
  thrust::device_ptr<int> thrust_offset(d_offset);

  if (maxpage == 1) {
    thrust::exclusive_scan(thrust_numneigh, thrust_numneigh + nparticles, thrust_offset);
    cudaMemset(d_pageidx, 0, d_pages_size);
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
  }

#if PARANOID
  check_decode(numneigh, firstneigh, pages);
#endif
}

void CudaNeighList::check_decode(int *numneigh, int **firstneigh, int **pages) {
  if (maxpage == 1) {
    // simulate segmented scan
    int *expected_offset = new int[nparticles];
    exclusive_scan_host(expected_offset, numneigh, nparticles);
    // check
    int *offset = new int[nparticles];
    cudaMemcpy(offset, d_offset, d_offset_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<nparticles; i++) {
      assert(offset[i] == expected_offset[i]);
    }

    // end-to-end check of decode
    int *neighidx = new int[maxpage*pgsize];
    cudaMemcpy(neighidx, d_neighidx, d_neighidx_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<nparticles; i++) {
      for (int j=0; j<numneigh[i]; j++) {
        int expected = firstneigh[i][j];
        int myoffset = offset[i];
        int actual = neighidx[myoffset+j];
        assert(expected == actual);
      }
    }

    delete[] expected_offset;
    delete[] neighidx;
  } else {
    // simulate decode_neighlist_p1
    int *expected_pagebreak = new int[nparticles];
    int *expected_pageidx = new int[nparticles];
    for (int i=0; i<nparticles; i++) {
      int *myfirstneigh = firstneigh[i];
      int mypagebreak = 0;
      int mypage = -1;
      for (int p=0; p<maxpage; p++) {
        mypagebreak |= (myfirstneigh == pages[p] ? 1 : 0);
        if ( (pages[p] <= myfirstneigh) &&
                         (myfirstneigh < (pages[p]+pgsize)) ) {
          mypage = p;
        }
      }
      expected_pagebreak[i] = mypagebreak;
      expected_pageidx[i] = mypage;
    }
    // check
    int *pageidx = new int[nparticles];
    cudaMemcpy(pageidx, d_pageidx, d_pageidx_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<nparticles; i++) {
      assert (pageidx[i] == expected_pageidx[i]);
    }

    // simulate segmented scan
    int *expected_offset = new int[nparticles];
    segmented_exclusive_scan_host(expected_offset, numneigh, expected_pagebreak, nparticles);
    // check
    int *offset = new int[nparticles];
    cudaMemcpy(offset, d_offset, d_offset_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<nparticles; i++) {
      assert(offset[i] == expected_offset[i]);
    }

    // end-to-end check of decode
    int *neighidx = new int[maxpage*pgsize];
    cudaMemcpy(neighidx, d_neighidx, d_neighidx_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<nparticles; i++) {
      for (int j=0; j<numneigh[i]; j++) {
        int expected = firstneigh[i][j];
        int mypage = pageidx[i];
        int myoffset = offset[i];
        int actual = neighidx[(mypage*pgsize)+myoffset+j];
        assert(expected == actual);
      }
    }

    delete[] expected_pagebreak;
    delete[] expected_pageidx;
    delete[] pageidx;
    delete[] expected_offset;
    delete[] offset;
    delete[] neighidx;
  }
}
