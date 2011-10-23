{% for c in consts -%}
  __constant__ {{ c.type }} {{ c.devname() }};
{% endfor %}

#include "{{ name }}_kernel.h"

