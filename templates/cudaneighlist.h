#ifndef {{ headername }}_CUDANEIGHLIST_H
#define {{ headername }}_CUDANEIGHLIST_H

#include "cudaneighlist.h"

class {{classname}}CudaNeighList : public CudaNeighList {
  private:
    {% for p in params if p.is_type('N', '-') -%}
      size_t {{ p.devname(suf='_size') }};
    {% endfor %}
  public:
    {% for p in params if p.is_type('N', '-') -%}
      {{ p.type }} {{ p.devname(pre='*') }};
    {% endfor %}

    {{classname}}CudaNeighList(int block_size, int nparticles, int maxpage, int pgsize);
    ~{{classname}}CudaNeighList();

    void reload(int *numneigh, int **firstneigh, int **pages, int reload_maxpage
      {%- for p in params if p.is_type('N', '-') -%}
        , {{ p.pages() }}
      {% endfor %}
    );

{% for p in params if p.is_type('N', '-') %}
    void load_{{ p.name() }}({{ p.pages() }});
    void unload_{{ p.name() }}({{ p.pages() }});
{% endfor %}

};

#endif

