#ifndef {{ headername }}_WRAPPER_H
#define {{ headername }}_WRAPPER_H

#include "clwrapper.h"
#include "{{ name }}_gpuneighlist.h"

class {{ classname }}Wrapper {
  private:
    // OpenCL parameters
    CLWrapper &clw;
    size_t wx;
    size_t gx;
    int N;

    // implementations
    cl_kernel tpa;
    cl_kernel bpa;
    
  public:
    {%- for p in params if p.is_type('P', '-') %}
    cl_mem {{ p.devname() }};
    {%- endfor %}
    {{ classname }}GpuNeighList *d_nl;

  public:
    {{ classname }}Wrapper(
      CLWrapper &clw, size_t wx,
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

    void run(
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

