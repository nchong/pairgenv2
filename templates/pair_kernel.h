#ifndef HOST_PAIR_KERNEL_H
#define HOST_PAIR_KERNEL_H
{% for c in consts -%}
  {{ c.type }} {{ c.devname() }};
{% endfor %}

#include "{{ name }}_kernel.h"

#endif

