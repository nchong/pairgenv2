#include "gpuneighlist.h"
#include "scan.h"
#include "scanref.h"
#include "segscan.h"

#include <cassert>

#define PARANOID true

GpuNeighList::GpuNeighList(CLWrapper &clw, size_t wx,
  int nparticles, int maxpage, int pgsize) :

  clw(clw), wx(wx),
  nparticles(nparticles), maxpage(maxpage), pgsize(pgsize),

  d_numneigh_size(nparticles * sizeof(int)),
  d_firstneigh_size(nparticles * sizeof(int *)),
  d_pages_size(maxpage * sizeof(int *)),
  d_pagebreak_size(d_numneigh_size),
  d_pageidx_size(d_numneigh_size),
  d_offset_size(d_numneigh_size),
  d_neighidx_size(maxpage * pgsize * sizeof(int)),

  scan(new Scan(clw, wx)),
  segscan(new SegmentedScan(clw, wx))
{

  d_numneigh = clw.dev_malloc(d_numneigh_size, CL_MEM_READ_ONLY);
  d_firstneigh = clw.dev_malloc(d_firstneigh_size, CL_MEM_READ_ONLY);
  d_pages = clw.dev_malloc(d_pages_size, CL_MEM_READ_ONLY);
  d_pagebreak = clw.dev_malloc(d_pagebreak_size);
  d_pageidx = clw.dev_malloc(d_pageidx_size);
  d_offset = clw.dev_malloc(d_offset_size);
  d_neighidx = clw.dev_malloc(d_neighidx_size, CL_MEM_READ_ONLY);

  string flags = sizeof(int *) == 4 ? " -D IS32BIT" :
                 sizeof(int *) == 8 ? " -D IS64BIT" : "";
#if EMBED_CL
  #include "decode.cl.h"
  clw.create_all_kernels(clw.compile_from_string((char *)&decode_cl, flags));
#else
  clw.create_all_kernels(clw.compile("decode.cl", flags));
#endif
  decode_neighlist_p1 = clw.kernel_of_name("decode_neighlist_p1");
}

GpuNeighList::~GpuNeighList() {
  clw.dev_free(d_numneigh);
  clw.dev_free(d_firstneigh);
  clw.dev_free(d_pages);
  clw.dev_free(d_pageidx);
  clw.dev_free(d_offset);
  clw.dev_free(d_neighidx);
}

/*
 * NB: does not update maxpage!
 */
void GpuNeighList::resize(int new_maxpage) {
  clw.dev_free(d_pages);
  clw.dev_free(d_neighidx);
  d_pages_size = new_maxpage * sizeof(int *);
  d_neighidx_size = new_maxpage * pgsize * sizeof(int);
  d_pages = clw.dev_malloc(d_pages_size, CL_MEM_READ_ONLY);
  d_neighidx = clw.dev_malloc(d_neighidx_size, CL_MEM_READ_ONLY);
}

void GpuNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage) {
  // nb: we do not expect nparticles or pgsize to change
  // resize if necessary
  if (maxpage < reload_maxpage) {
    resize(reload_maxpage);
    maxpage = reload_maxpage;
  }

  // copy into device memory
  clw.memcpy_to_dev(d_numneigh, d_numneigh_size, numneigh);
  clw.memcpy_to_dev(d_firstneigh, d_firstneigh_size, firstneigh);
  load_pages(d_neighidx, pages);
  clw.memcpy_to_dev(d_pages, d_pages_size, pages);

  // scan or segmented scan on this
  clw.memcpy_to_dev(d_offset, d_offset_size, numneigh);

  if (maxpage == 1) {
    // every particle has pageidx=0 (implicitly) 
    // just do scan on d_offset
    printf("Running scan...");
    scan->scan(d_offset, nparticles);
    printf("done\n");
  } else {
    // first part of decode: for each particle
    //   - which page their neighbor list resides in (pageidx)
    //   - if they are the first
    size_t wx = 1;
    // number of workgroups of size [wx] workitems
    int k = ((nparticles / wx)+1);
    size_t gx = (size_t) k * wx;
    int _pgsize = pgsize * sizeof(int); // in chars
    clw.kernel_arg(decode_neighlist_p1,
      nparticles,
      d_firstneigh,
      maxpage,
      d_pages,
      _pgsize,
      d_pagebreak,
      d_pageidx);
    printf("Running decode_neighlist_p1...");
    float k0 = clw.run_kernel_with_timing(decode_neighlist_p1, /*dim=*/1, &gx, &wx);
    printf("done (%fms)\n", k0);

    printf("Running segmented scan...");
    segscan->scan(d_offset, d_pagebreak, nparticles);
    printf("done\n");
  }

#if PARANOID
    printf("Running checks...");
    check_decode(numneigh, firstneigh, pages);
    printf("done\n");
#endif
}

void GpuNeighList::check_decode(int *numneigh, int **firstneigh, int **pages) {
  if (maxpage == 1) {
    // simulate segmented scan
    int *expected_offset = new int[nparticles];
    exclusive_scan_host(expected_offset, numneigh, nparticles);
    // check
    int *offset = new int[nparticles];
    clw.memcpy_from_dev(d_offset, d_offset_size, offset);
    for (int i=0; i<nparticles; i++) {
      assert(offset[i] == expected_offset[i]);
    }

    // end-to-end check of decode
    int *neighidx = new int[maxpage*pgsize];
    clw.memcpy_from_dev(d_neighidx, d_neighidx_size, neighidx);
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
    int *pagebreak = new int[nparticles];
    int *pageidx = new int[nparticles];
    clw.memcpy_from_dev(d_pagebreak, d_pagebreak_size, pagebreak);
    clw.memcpy_from_dev(d_pageidx, d_pageidx_size, pageidx);
    for (int i=0; i<nparticles; i++) {
      assert (pagebreak[i] == expected_pagebreak[i]);
      assert (pageidx[i] == expected_pageidx[i]);
    }

    // simulate segmented scan
    int *expected_offset = new int[nparticles];
    segmented_exclusive_scan_host(expected_offset, numneigh, expected_pagebreak, nparticles);
    // check
    int *offset = new int[nparticles];
    clw.memcpy_from_dev(d_offset, d_offset_size, offset);
    for (int i=0; i<nparticles; i++) {
      assert(offset[i] == expected_offset[i]);
    }

    // end-to-end check of decode
    int *neighidx = new int[maxpage*pgsize];
    clw.memcpy_from_dev(d_neighidx, d_neighidx_size, neighidx);
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
    delete[] pagebreak;
    delete[] pageidx;
    delete[] expected_offset;
    delete[] offset;
    delete[] neighidx;
  }
}
