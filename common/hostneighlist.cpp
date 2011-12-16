#include "hostneighlist.h"
#include "scanref.h"

int *host_decode_neighlist(
  int nparticles,
  int maxpage,
  int *numneigh, int **firstneigh,
  int **pages, int pgsize) {
  int *h_offset = new int[nparticles];
  if (maxpage == 1) {
    exclusive_scan_host(h_offset, numneigh, nparticles);
  } else {
    // simulate decode_neighlist_p1
    int *h_pagebreak = new int[nparticles];
    int *h_pageidx = new int[nparticles];
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
      h_pagebreak[i] = mypagebreak;
      h_pageidx[i] = mypage;
    }
    // simulate segmented scan
    segmented_exclusive_scan_host(h_offset, numneigh, h_pagebreak, nparticles);

    // simulate second part of decode
    for (int i=0; i<nparticles; i++) {
      h_offset[i] += h_pageidx[i]*pgsize;
    }

    delete[] h_pagebreak;
    delete[] h_pageidx;
  }
  return h_offset;
}
