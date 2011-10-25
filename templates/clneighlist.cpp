#include "{{name}}_clneighlist.h"
#include "clwrapper.h"
#include "clerror.h"

{{classname}}CLNeighList::{{classname}}CLNeighList(CLWrapper &clw, size_t wx, int nparticles, int maxpage, int pgsize) : 
  CLNeighList(clw, wx, nparticles, maxpage, pgsize) {
  {% for p in params if p.is_type('N', '-') -%}
  {{ p.devname(suf='_size') }} = {{ p.sizeof() }};
  {% endfor %}
  {% for p in params if p.is_type('N', '-') -%}
  {{ p.devname() }} = clw.dev_malloc({{ p.devname(suf='_size') }});
  {% endfor %}
}

{{classname}}CLNeighList::~{{classname}}CLNeighList() {
  {% for p in params if p.is_type('N', '-') -%}
  clw.dev_free({{ p.devname() }});
  {% endfor %}
}

void {{classname}}CLNeighList::reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage
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
    {{ p.devname() }} = clw.dev_malloc({{ p.devname(suf='_size') }});
    {% endfor %}
    maxpage = reload_maxpage;
  }
  CLNeighList::reload(numneigh, firstneigh, pages, maxpage);
  {% for p in params if p.is_type('N', '-') -%}
  load_pages({{ p.devname() }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
  {% endfor %}
}

{% for p in params if p.is_type('N', '-') %}
void {{classname}}CLNeighList::unload_{{ p.name() }}({{ p.pages() }}) {
  unload_pages({{ p.name(pre='d_') }}, {{ p.name(pre='h_', suf='pages') }}, /*dim=*/{{ p.dim }});
}
{% endfor %}

