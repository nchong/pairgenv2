#include "{{name}}_cudaneighlist.h"

{{classname}}CudaNeighList::{{classname}}CudaNeighList(int block_size, int nparticles, int maxpage, int pgsize) : 
  CudaNeighList(block_size, nparticles, maxpage, pgsize)
  {%- for p in params if p.is_type('N', '-') -%}{{- ',' if loop.first }}
  {{ p.devname(suf='_size') }}({{ p.sizeof() }}){{ ', ' if not loop.last }}
  {%- endfor %} {
  {% for p in params if p.is_type('N', '-') -%}
  cudaMalloc((void **)&{{ p.devname() }}, {{ p.devname(suf='_size') }});
  {% endfor %}
}

{{classname}}CudaNeighList::~{{classname}}CudaNeighList() {
  {% for p in params if p.is_type('N', '-') -%}
  cudaFree({{ p.devname() }});
  {% endfor %}
}

void {{classname}}CudaNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage
  {% for p in params if p.is_type('N', '-') -%}
    , {{ p.pages() }}
  {% endfor %}
  ) {
  if (maxpage < reload_maxpage) {
    resize(reload_maxpage);
    {% for p in params if p.is_type('N', '-') -%}
    cudaFree({{ p.devname() }});
    {% endfor %}
    {% for p in params if p.is_type('N', '-') -%}
    {{ p.devname(suf='_size') }} = reload_maxpage*pgsize*{{ p.dim }}*sizeof({{ p.type }});
    {% endfor %}
    {% for p in params if p.is_type('N', '-') -%}
    cudaMalloc((void **)&{{ p.devname() }}, {{ p.devname(suf='_size') }});
    {% endfor %}
    maxpage = reload_maxpage;
  }
  CudaNeighList::reload(numneigh, firstneigh, pages, maxpage);
  {% for p in params if p.is_type('N', '-') -%}
  load_pages({{ p.devname() }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
  {% endfor %}
}

{% for p in params if p.is_type('N', '-') %}
void {{classname}}CudaNeighList::load_{{ p.name() }}({{ p.pages() }}) {
  load_pages({{ p.name(pre='d_') }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
}
void {{classname}}CudaNeighList::unload_{{ p.name() }}({{ p.pages() }}) {
  unload_pages({{ p.name(pre='d_') }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
}
{% endfor %}

