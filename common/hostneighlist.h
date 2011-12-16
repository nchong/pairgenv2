#ifndef HOSTNEIGHLIST_H
#define HOSTNEIGHLIST_H

/*
 * Host-side neighbor list decode.
 *
 * This is a fallback for the CUDA/OpenCL specific neighbor list decode procedures.
 *
 * This function creates an int array h_offset of length nparticles.
 * h_offset[i] is an integer offset pointing to the start of particle i's neighbor list within d_neighidx array (see the load_pages procedure).
 *
 * h_offset should be copied into device memory and then disposed.
 */
int *host_decode_neighlist(
  int nparticles,
  int maxpage,
  int *numneigh, int **firstneigh,
  int **pages, int pgsize);

#endif
