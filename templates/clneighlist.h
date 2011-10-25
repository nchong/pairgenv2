#ifndef {{ headername }}_CLNEIGHLIST_H
#define {{ headername }}_CLNEIGHLIST_H

#include "clneighlist.h"

class {{classname}}CLNeighList : public CLNeighList {
  private:
    {% for p in params if p.is_type('N', '-') -%}
      size_t {{ p.devname(suf='_size') }};
    {% endfor %}
  public:
    {% for p in params if p.is_type('N', '-') -%}
      cl_mem {{ p.devname() }};
    {% endfor %}

    {{classname}}CLNeighList(CLWrapper &clw, size_t wx, int nparticles, int maxpage, int pgsize);
    ~{{classname}}CLNeighList();

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

