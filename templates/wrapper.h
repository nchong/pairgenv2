#ifndef {{ headername }}_WRAPPER_H
#define {{ headername }}_WRAPPER_H

#include "clwrapper.h"
#include "{{ name }}_clneighlist.h"

#include <vector>
double get_m0(); std::vector<double> &get_m0_raw();
double get_k0(); std::vector<double> &get_k0_raw();
double get_m1(); std::vector<double> &get_m1_raw();

class {{ classname }}Wrapper {
  private:
    // OpenCL parameters
    CLWrapper &clw;
    int N;
    size_t wx;
    size_t tpa_gx;
    size_t bpa_gx;

    // decompositions
    cl_kernel tpa;
    cl_kernel bpa;
    
  public:
    {%- for p in params if p.is_type('P', '-') %}
    cl_mem {{ p.devname() }};
    {%- endfor %}
    {{ classname }}CLNeighList *d_nl;

  public:
    {{ classname }}Wrapper(
      CLWrapper &clw, size_t wx, const char *flags,
      int N, int maxpage, int pgsize,
      {% for c in consts -%}
        {{ c.decl(pre='h_', include_dim=False) }},
      {% endfor -%}
      {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
        {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
      {% endfor -%}
    );

    ~{{ classname }}Wrapper();

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

