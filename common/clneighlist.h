#ifndef CLNEIGHLIST_H
#define CLNEIGHLIST_H

#include "clwrapper.h"
#include "scan.h"
#include "segscan.h"

/*
 * OpenCL Neighbor List manager
 */
class CLNeighList {
  protected:
    //opencl wrapper
    CLWrapper &clw;
    //workgroup size
    size_t wx;
  public:
    //these parameters control the size of the neighbor list
    int nparticles;
    int maxpage;
    int pgsize;
  protected:
    //sizes of neighbor list structures
    size_t d_numneigh_size;
    size_t d_firstneigh_size;
    size_t d_pages_size;
    size_t d_pagebreak_size;
    size_t d_pageidx_size;
    size_t d_offset_size;
    size_t d_neighidx_size;
  public:
    //device neighbor list structures
    cl_mem d_numneigh;
    cl_mem d_firstneigh;
    cl_mem d_pages;
    cl_mem d_pagebreak;
    cl_mem d_pageidx;
    cl_mem d_offset;
    cl_mem d_neighidx;
  private:
    //kernels
    cl_kernel decode_neighlist_p1;
    cl_kernel decode_neighlist_p2;
    //scans
    Scan *scan;
    SegmentedScan *segscan;
    //timings
    float tload; float tunload;
    float tdecode;

  protected:
    template <class T>
    void load_pages(cl_mem d_mem, T **h_ptr, int dim=1) {
      size_t size = pgsize*sizeof(T)*dim;
      for (int p=0; p<maxpage; p++) {
        size_t offset = p*size;
        tload += clw.memcpy_to_dev(d_mem, size, h_ptr[p], offset);
      }
    }
    template <class T>
    void unload_pages(cl_mem d_mem, T **h_ptr, int dim=1) {
      size_t size = pgsize*sizeof(T)*dim;
      for (int p=0; p<maxpage; p++) {
        size_t offset = p*size;
        tunload += clw.memcpy_from_dev(d_mem, size, h_ptr[p], offset);
      }
    }
    void resize(int new_maxpage);

  private:
    void check_decode(int *numneigh, int **firstneigh);

  public:
    CLNeighList(CLWrapper &clw, size_t wx, int nparticles, int maxpage, int pgsize);
    ~CLNeighList();
    void reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage);
    int get_maxpage();
    void get_timers(map<string,float> &timings);
};

#endif
