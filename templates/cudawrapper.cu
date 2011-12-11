{%- macro memcpy_to_dev_args(p) -%}
  {{- p.devname() -}}, 
  {{- p.name(pre='h_') -}},
  {{- p.sizeof() -}},
  cudaMemcpyHostToDevice
{%- endmacro -%}
{%- macro memcpy_from_dev_args(p) -%}
  {{- p.name(pre='h_') -}},
  {{- p.devname() -}}, 
  {{- p.sizeof() -}},
  cudaMemcpyDeviceToHost
{%- endmacro -%}
#include "{{ name }}_cudawrapper.h"
#include "{{ name }}_cudaneighlist.h"
#include "{{ name }}_tpa.cu"
#include "{{ name }}_bpa.cu"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <iostream>

using namespace std;

#include "posix_timer.h"
#include <vector>
static SimpleTimer m0; static vector<double> m0_raw;
static SimpleTimer k0; static vector<double> k0_raw;
static SimpleTimer m1; static vector<double> m1_raw;
double get_cuda_m0() { return m0.total_time(); }
double get_cuda_k0() { return k0.total_time(); }
double get_cuda_m1() { return m1.total_time(); }
vector<double> &get_cuda_m0_raw() { return m0_raw; }
vector<double> &get_cuda_k0_raw() { return k0_raw; }
vector<double> &get_cuda_m1_raw() { return m1_raw; }

#ifdef TRACE
#warning Turning TRACE on will affect timing results!
#include "cuPrintf.cu"
#endif

#define MAX_GRID_DIM 65535

{{ classname }}CudaWrapper::{{ classname }}CudaWrapper(
    int block_size,
    int N, int maxpage, int pgsize,
    {% for c in consts -%}
      {{ c.decl(pre='h_', include_dim=False) }},
    {% endfor -%}
    {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {% endfor -%}
  ) :
  block_size(block_size),
  N(N),
  tpa_grid_size((N/block_size)+1),
  bpa_grid_size(min(N, MAX_GRID_DIM), max((int)ceil((float)N/MAX_GRID_DIM),1)),
  // size of per-block array for
  {%- for p in params if p.is_type('P', 'SUM') -%}
    {{ ' ' }}{{ p.devname() }}{{ ',' if not loop.last }}
  {%- endfor %}
  bpa_shared_mem_size(
    {%- for p in params if p.is_type('P', 'SUM') -%}
      (block_size*{{ p.dim }}*sizeof({{ p.type }})){{ ' + ' if not loop.last }}
    {%- endfor -%}
  ),
  d_nl(new {{ classname }}CudaNeighList(block_size, N, maxpage, pgsize))
{
  {% for c in consts -%}
    cudaMemcpyToSymbol("{{ c.devname() }}", &{{ c.name(pre='h_') }}, {{ c.sizeof() }}, 0, cudaMemcpyHostToDevice);
  {% endfor %}
  {% for p in params if p.is_type('P', '-') -%}
    cudaMalloc((void **)&{{ p.devname() }}, {{ p.sizeof() }});
  {% endfor %}
  {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
    cudaMemcpy({{ memcpy_to_dev_args(p) }});
  {% endfor %}
#if DEBUG
  cerr << "[DEBUG] N=" << N << endl;
  cerr << "[DEBUG] Kernel TpA parameters grid_size=" << tpa_grid_size << " block_size=" << block_size << endl;
  cerr << "[DEBUG] Kernel BpA parameters grid_size=(" << bpa_grid_size.x << ", " << bpa_grid_size.y << ") block_size=" << block_size << endl;
#endif
  assert(bpa_grid_size.y < MAX_GRID_DIM);
}

{{ classname }}CudaWrapper::~{{ classname }}CudaWrapper() {
  {% for p in params if not p.is_type('N', '-') -%}
    cudaFree({{ p.devname() }});
  {% endfor %}
  delete(d_nl);
}

void {{ classname }}CudaWrapper::refill_neighlist(
  int *h_numneigh,
  int **h_firstneigh,
  int **h_pages,
  int maxpage,
  {% for p in params if p.is_type('N', '-') -%}
    {{ p.pages() }}{{ ',' if not loop.last }}
  {% endfor -%}
) {
  d_nl->reload(h_numneigh, h_firstneigh, h_pages, maxpage,
      {%- for p in params if p.is_type('N', '-') -%}
        {{ p.name(pre='h_',suf='pages') }}{{ ', ' if not loop.last }}
      {%- endfor -%}
  );
}

void {{ classname }}CudaWrapper::run(
  kernel_decomposition kernel,
  {% for p in params -%}
    {%- if p.is_type('P', 'RO') and p.reload -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {%- elif p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {%- elif p.is_type('N', 'RO') and p.reload -%}
      {{ p.pages() }}{{ ',' if not loop.last }}
    {%- elif p.is_type('N', 'RW') -%}
      {{ p.pages() }}{{ ',' if not loop.last }}
    {%- else -%}
      // {{ p.name() }} is not reloaded
    {%- endif %}
  {% endfor -%}
) {
  m0.start();
  {% for p in params if p.is_type('P', '-') and p.reload -%}
    cudaMemcpy({{ memcpy_to_dev_args(p) }});
  {% endfor -%}
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    cudaMemcpy({{ memcpy_to_dev_args(p) }});
  {% endfor %}
  m0_raw.push_back(m0.stop_and_add_to_total());

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pre-compute-kernel error: %s.\n", cudaGetErrorString(err));
    exit(1);
  }
#ifdef TRACE
  cudaPrintfInit();
#endif
  k0.start();
  if (kernel == TPA) {
    {{ name }}_tpa<<<tpa_grid_size, block_size>>>(
      N,
      {% for p in params if p.is_type('P', 'RO') -%}
      {{ p.devname() }},
      {% endfor -%}
      d_nl->d_numneigh, d_nl->d_offset, d_nl->d_neighidx,
      {% for p in params if not p.is_type('P', 'RO') -%}
        {%- if p.is_type('N', '-') -%}
      d_nl->{{ p.devname() }}{{ ', ' if not loop.last }}
        {%- else -%}
      {{ p.devname() }}{{ ', ' if not loop.last }}
        {%- endif -%}
      {% endfor -%}
    );
  } else if (kernel == BPA) {
    {{ name }}_bpa<<<bpa_grid_size, block_size, bpa_shared_mem_size>>>(
      N,
      {% for p in params if p.is_type('P', 'RO') -%}
      {{ p.devname() }},
      {% endfor -%}
      d_nl->d_numneigh, d_nl->d_offset, d_nl->d_neighidx,
      {%- for p in params if not p.is_type('P', 'RO') %}
        {%- if p.is_type('N', '-') %}
      d_nl->{{ p.devname() }}{{ ', ' if not loop.last }}
        {%- else %}
      {{ p.devname() }}{{ ', ' if not loop.last }}
        {%- endif -%}
      {% endfor -%}
    );
  }
  cudaThreadSynchronize();
  k0_raw.push_back(k0.stop_and_add_to_total());
#ifdef TRACE
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Post-compute-kernel error: %s.\n", cudaGetErrorString(err));
    exit(1);
  }

  m1.start();
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    cudaMemcpy({{ memcpy_from_dev_args(p) }});
  {% endfor %}

  {% for p in params if p.is_type('N', 'RW') -%}
  if ({{ p.name(pre='h_',suf='pages') }} != NULL) {
    d_nl->unload_{{ p.name() }}({{ p.name(pre='h_',suf='pages') }});
  }
  {% endfor %}
  m1_raw.push_back(m1.stop_and_add_to_total());
}

