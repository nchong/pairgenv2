#include "{{name}}_gpuneighlist.h"
#include "clwrapper.h"
#include "clerror.h"

{{classname}}GpuNeighList::{{classname}}GpuNeighList(CLWrapper &clw, size_t wx, int nparticles, int maxpage, int pgsize) : 
  GpuNeighList(clw, wx, nparticles, maxpage, pgsize) {
  {% for p in params if p.is_type('N', '-') -%}
  {{ p.devname(suf='_size') }} = {{ p.sizeof() }};
  {% endfor %}
  {% for p in params if p.is_type('N', '-') -%}
  {{ p.devname() }} = clw.dev_malloc({{ p.devname(suf='_size') }});
  {% endfor %}
}

{{classname}}GpuNeighList::~{{classname}}GpuNeighList() {
  {% for p in params if p.is_type('N', '-') -%}
  clw.dev_free({{ p.devname() }});
  {% endfor %}
}

void {{classname}}GpuNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage
  {% for p in params if p.is_type('N', '-') -%}
    , {{ p.pages() }}
  {% endfor %}
  ) {
  if (maxpage < reload_maxpage) {
    resize(reload_maxpage);
    {% for p in params if p.is_type('N', '-') -%}
    clw.dev_free({{ p.devname() }});
    {% endfor %}
    {% for p in params if p.is_type('N', '-') -%}
    {{ p.devname(suf='_size') }} = reload_maxpage*pgsize*{{ p.dim }}*sizeof({{ p.type }});
  {% endfor %}
    {% for p in params if p.is_type('N', '-') -%}
    {{ p.name(pre='d_') }} = clw.dev_malloc({{ p.devname(suf='_size') }});
    {% endfor %}
    maxpage = reload_maxpage;
  }
  GpuNeighList::reload(numneigh, firstneigh, pages, maxpage);
  {% for p in params if p.is_type('N', '-') -%}
  load_pages({{ p.devname() }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
  {% endfor %}
}

{% for p in params if p.is_type('N', '-') %}
void {{classname}}GpuNeighList::unload_{{ p.name() }}({{ p.pages() }}) {
  unload_pages({{ p.name(pre='d_') }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
}
{% endfor %}

