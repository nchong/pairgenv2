#ifndef CUDANEIGHLIST_H
#define CUDANEIGHLIST_H

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <map>
#include <string>
using namespace std;

/*
 * CUDA Neighbor List manager
 */
class CudaNeighList {
  public:
    //these parameters control the size of the neighbor list
    int nparticles;
    int maxpage;
    int pgsize;
  protected:
    //block size
    int block_size;
    dim3 grid_size;
#if TPN
    dim3 neighbor_grid_size;
#endif
  protected:
    //sizes of neighbor list structures
    size_t d_numneigh_size;
    size_t d_firstneigh_size;
    size_t d_pages_size;
    size_t d_pageidx_size;
    size_t d_offset_size;
    size_t d_neighidx_size;
#if TPN
    size_t d_valid_size;
    size_t d_dati_size;
    size_t d_tad_size;
    size_t d_ffo_size;
    size_t d_nel_size;
#endif
  public:
    //device neighbor list structures
    int  *d_numneigh;
    int **d_firstneigh;
    int **d_pages;
    int  *d_pageidx;
    int  *d_offset;
    int  *d_neighidx;
#if TPN
    int  *d_valid;
    int  *d_dati;
    int  *d_tad;
    int  *d_ffo;
    int  *d_nel;
#endif
  private:
    //timings
    float tload; float tunload;
    float tdecode;

  protected:
    template <typename T>
    void load_pages(T *d_ptr, T **h_ptr, int dim=1) {
      size_t size = pgsize*sizeof(T)*dim;
      for (int p=0; p<maxpage; p++) {
        cudaMemcpy(&(d_ptr[p*pgsize*dim]), h_ptr[p], size, cudaMemcpyHostToDevice);
      }
    }
    template <typename T>
    void unload_pages(T *d_ptr, T **h_ptr, int dim=1) {
      size_t size = pgsize*sizeof(T)*dim;
      for (int p=0; p<maxpage; p++) {
        cudaMemcpy(h_ptr[p], &(d_ptr[p*pgsize*dim]), size, cudaMemcpyDeviceToHost);
      }
    }
    void resize(int new_maxpage);

  private:
    void check_decode(int *numneigh, int **firstneigh);
#if TPN
    void reload_inverse();
    void check_inverse();
#endif

  public:
    CudaNeighList(int block_size,
                  int nparticles, int maxpage, int pgsize);
    virtual ~CudaNeighList();
    void reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage);
    int get_maxpage();
    void get_timers(map<string,float> &timings);
};

#endif
