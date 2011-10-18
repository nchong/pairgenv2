#ifndef {{ headername }}_GPUNEIGHLIST_H
#define {{ headername }}_GPUNEIGHLIST_H

#include "gpuneighlist.h"

class {{classname}}GpuNeighList : public GpuNeighList {
  private:
    {% for p in params if p.is_type('N', '-') -%}
      size_t {{ p.devname(suf='_size') }};
    {% endfor %}
  public:
    {% for p in params if p.is_type('N', '-') -%}
      cl_mem {{ p.devname() }};
    {% endfor %}

    {{classname}}GpuNeighList(CLWrapper &clw, size_t wx, int nparticles, int maxpage, int pgsize);
    ~{{classname}}GpuNeighList();

    void reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage
      {%- for p in params if p.is_type('N', '-') -%}
        , {{ p.pages() }}
      {% endfor %}
    );

{% for p in params if p.is_type('N', '-') %}
    void unload_{{ p.name() }}({{ p.pages() }});
{% endfor %}

};

#endif
