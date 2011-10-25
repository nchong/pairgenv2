#ifndef {{ headername }}_PAIR_KERNEL_CU
#define {{ headername }}_PAIR_KERNEL_CU

{% for c in consts -%}
  __constant__ {{ c.type }} {{ c.devname() }};
{% endfor %}

#include "{{ name }}_kernel.h"

#endif

