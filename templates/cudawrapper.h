#ifndef {{ headername }}_CUDAWRAPPER_H
#define {{ headername }}_CUDAWRAPPER_H

#include "{{ name }}_cudaneighlist.h"

#include <vector>
double get_cuda_m0(); std::vector<double> &get_cuda_m0_raw();
double get_cuda_k0(); std::vector<double> &get_cuda_k0_raw();
double get_cuda_m1(); std::vector<double> &get_cuda_m1_raw();

class {{ classname }}CudaWrapper {
  private:
    int block_size;
    int N;
    int tpa_grid_size;
    int bpa_grid_x_size;
    int bpa_grid_y_size;
    dim3 bpa_grid_size;
    size_t bpa_shared_mem_size;

  public:
    {%- for p in params if p.is_type('P', '-') %}
    {{ p.type }} {{ p.devname(pre='*') }};
    {%- endfor %}
    {{ classname }}CudaNeighList *d_nl;

  public:
    {{ classname }}CudaWrapper(
      int block_size,
      int N, int maxpage, int pgsize,
      {% for c in consts -%}
        {{ c.decl(pre='h_', include_dim=False) }},
      {% endfor -%}
      {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
        {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
      {% endfor -%}
    );

    ~{{ classname }}CudaWrapper();

    void refill_neighlist(
      int *h_numneigh,
      int **h_firstneigh,
      int **h_pages,
      int maxpage,
      {% for p in params if p.is_type('N', '-') -%}
        {{ p.pages() }}{{ ',' if not loop.last }}
      {% endfor -%}
    );

    enum kernel_decomposition { TPA, BPA };
    void run(
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
    );

};
#endif

