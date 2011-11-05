/*
 * For each particle (tid) determine which page of the CPU neighbor list
 * contains the neighbor list for this particle.
 */
#ifdef IS32BIT
  #define HOSTPTR_MOCK_TYPE unsigned int
#elif IS64BIT
  #define HOSTPTR_MOCK_TYPE unsigned long
#else
  #error Can only deal with 32 or 64-bit machines
#endif

__kernel void decode_neighlist_p1(
  //inputs
    int nparticles,
    __global HOSTPTR_MOCK_TYPE *firstneigh,
    int maxpage,
    __global HOSTPTR_MOCK_TYPE *pages,
    int pgsize,
  //outputs
    __global int *pagebreak,
    __global int *pageidx
) {
  int tid = get_global_id(0);
  if (tid < nparticles) {
    HOSTPTR_MOCK_TYPE myfirstneigh = firstneigh[tid];
    int mypagebreak = 0;
    int mypage = -1;
    for (int p=0; p<maxpage; p++) {
      mypagebreak |= myfirstneigh == pages[p] ? 1 : 0;
      if ( (pages[p] <= myfirstneigh) &&
                       (myfirstneigh < (pages[p]+pgsize)) ) {
        mypage = p;
      }
    }
    pagebreak[tid] = mypagebreak;
    pageidx[tid] = mypage;
  }
}

__kernel void decode_neighlist_p2(
  //input
  int nparticles,
  __global int *pageidx,
  __global int pgsize,
  //inout
  __global int *offset
) {
  int tid = get_global_id(0);
  if (tid < nparticles) {
    int mypage = pageidx[tid];
    offset[tid] += (mypage*pgsize);
  }
}
