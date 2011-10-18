{%- macro memcpy_args(p) -%}
  {{- p.devname() -}}, 
  {{- p.sizeof() -}}, 
  {{- p.name(pre='h_') -}}
{%- endmacro -%}
#include "{{ name }}_wrapper.h"
#include "{{ name }}_gpuneighlist.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <sstream>

using namespace std;

{{ classname }}Wrapper::{{ classname }}Wrapper(
    CLWrapper &clw, size_t wx,
    int N, int maxpage, int pgsize,
    {% for c in consts -%}
      {{ c.decl(pre='h_', include_dim=False) }},
    {% endfor -%}
    {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {% endfor -%}
  ) :
  clw(clw), wx(wx), N(N) {
    gx = wx * ((N/wx)+1);
    d_nl = new {{ classname }}GpuNeighList(clw, wx, N, maxpage, pgsize);

    {% for p in params if p.is_type('P', '-') -%}
    {{ p.devname() }} = clw.dev_malloc({{ p.sizeof() }});
    {% endfor %}
    {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
      clw.memcpy_to_dev({{ memcpy_args(p) }});
    {% endfor %}
    stringstream extra_flags;
    extra_flags << " -I .";
    {% for c in consts -%}
    extra_flags << " -D {{ c.hashdefine() }}=" << {{ c.name(pre='h_')}};
    {% endfor %}
    clw.create_all_kernels(clw.compile("{{name}}_tpa_compute_kernel.cl", extra_flags.str()));
    tpa = clw.kernel_of_name("{{name}}_tpa_compute_kernel");
  //clw.create_all_kernels(clw.compile("{{name}}_bpa_compute_kernel.cl"));
  //bpa = clw.kernel_of_name("{{name}}_bpa_compute_kernel");
}

{{ classname }}Wrapper::~{{ classname }}Wrapper() {
  {% for p in params if not p.is_type('N', '-') -%}
    clw.dev_free({{ p.devname() }});
  {% endfor %}
  delete(d_nl);
}

void {{ classname }}Wrapper::refill_neighlist(
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

void {{ classname }}Wrapper::run(
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
  {% for p in params if p.is_type('P', '-') and p.reload -%}
    clw.memcpy_to_dev({{ memcpy_args(p) }});
  {% endfor -%}
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    clw.memcpy_to_dev({{ memcpy_args(p) }});
  {% endfor %}
  clw.kernel_arg(tpa,
    N,
    {% for p in params if p.is_type('P', 'RO') -%}
    {{ p.devname() }},
    {% endfor -%}
    d_nl->d_numneigh, d_nl->d_pageidx, d_nl->d_offset, d_nl->pgsize, d_nl->d_neighidx,
    {% for p in params if not p.is_type('P', 'RO') -%}
      {%- if p.is_type('N', '-') -%}
    d_nl->{{ p.devname() }}{{ ', ' if not loop.last }}
      {%- else -%}
    {{ p.devname() }}{{ ', ' if not loop.last }}
      {%- endif -%}
    {% endfor -%}
  );
  clw.run_kernel(tpa, /*dim=*/1, &gx, &wx);
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    clw.memcpy_from_dev({{ memcpy_args(p) }});
  {% endfor %}

  {% for p in params if p.is_type('N', 'RW') -%}
  if ({{ p.name(pre='h_',suf='pages') }} != NULL) {
    d_nl->unload_{{ p.name() }}({{ p.name(pre='h_',suf='pages') }});
  }
  {% endfor %}
}

