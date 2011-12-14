#include "clneighlist.h"
#include "scan.h"
#include "segscan.h"
#include "hostneighlist.h"

#include <cassert>

#define ONLY_HOST true
#define PARANOID false

CLNeighList::CLNeighList(CLWrapper &clw, size_t wx,
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
  segscan(new SegmentedScan(clw, wx)),

  tload(0), tunload(0), tdecode(0)
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
  decode_neighlist_p2 = clw.kernel_of_name("decode_neighlist_p2");
}

CLNeighList::~CLNeighList() {
  clw.dev_free(d_numneigh);
  clw.dev_free(d_firstneigh);
  clw.dev_free(d_pages);
  clw.dev_free(d_pageidx);
  clw.dev_free(d_offset);
  clw.dev_free(d_neighidx);
}

void CLNeighList::get_timers(map<string,float> &timings) {
  if (clw.has_profiling()) {
    timings.insert(make_pair("GPUNL1. load pages   ", tload));
    timings.insert(make_pair("GPUNL2. unload pages ", tunload));
    timings.insert(make_pair("GPUNL2. decode_neighlist_p1", tdecode));
    scan->get_timers(timings);
    segscan->get_timers(timings);
  }
}

/*
 * NB: does not update maxpage!
 */
void CLNeighList::resize(int new_maxpage) {
  clw.dev_free(d_pages);
  clw.dev_free(d_neighidx);
  d_pages_size = new_maxpage * sizeof(int *);
  d_neighidx_size = new_maxpage * pgsize * sizeof(int);
  d_pages = clw.dev_malloc(d_pages_size, CL_MEM_READ_ONLY);
  d_neighidx = clw.dev_malloc(d_neighidx_size, CL_MEM_READ_ONLY);
}

void CLNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage) {
  // nb: we do not expect nparticles or pgsize to change
  // resize if necessary
  if (maxpage < reload_maxpage) {
    resize(reload_maxpage);
    maxpage = reload_maxpage;
  }

  // just copy directly into device memory
  clw.memcpy_to_dev(d_numneigh, d_numneigh_size, numneigh);
  load_pages(d_neighidx, pages);

#if ONLY_HOST
  int *h_offset = host_decode_neighlist(nparticles, maxpage, numneigh, firstneigh, pages, pgsize);
  clw.memcpy_to_dev(d_offset, d_offset_size, h_offset);
  delete[] h_offset;
#else
  // our aim is to construct, for each particle i, an offset into d_neighidx that is the start of the list of n neighbors for i
  clw.memcpy_to_dev(d_offset, d_offset_size, numneigh);
  if (maxpage == 1) {
    // just do scan on d_offset
    scan->scan(d_offset, nparticles);
  } else {
    // First part of decode. For each particle determine:
    //   - which page their list of neighbors resides in (pageidx)
    //   - if their list of neighbors is the first entry of a page (pagebreak)
    clw.memcpy_to_dev(d_firstneigh, d_firstneigh_size, firstneigh);
    clw.memcpy_to_dev(d_pages, d_pages_size, pages);
    // number of workgroups of size [wx] workitems
    int k = ((nparticles / wx)+1);
    size_t gx = (size_t) k * wx;
    // necessary because we send pointer data in firstneigh and pages as 32 or 64 bit ints
    int pgsize_in_chars = pgsize * sizeof(int);
    clw.kernel_arg(decode_neighlist_p1,
      nparticles,
      d_firstneigh,
      maxpage,
      d_pages,
      pgsize_in_chars,
      d_pagebreak,
      d_pageidx);
    tdecode += clw.run_kernel_with_timing(decode_neighlist_p1, /*dim=*/1, &gx, &wx);

    // compute offset within each page
    segscan->scan(d_offset, d_pagebreak, nparticles);

    // Second part of decode. Uniformly add the page offset
    clw.kernel_arg(decode_neighlist_p2,
      nparticles,
      d_pageidx,
      pgsize,
      d_offset);
    tdecode += clw.run_kernel_with_timing(decode_neighlist_p2, /*dim=*/1, &gx, &wx);
  }
#endif

#if PARANOID
    check_decode(numneigh, firstneigh);
#endif
}

/* end to end check of decode */
void CLNeighList::check_decode(int *numneigh, int **firstneigh) {
  int *neighidx = new int[maxpage*pgsize];
  int *offset = new int[nparticles];
  clw.memcpy_from_dev(d_neighidx, d_neighidx_size, neighidx);
  clw.memcpy_from_dev(d_offset, d_offset_size, offset);
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
