{% if cl_khr_fp64 %}
#ifndef cl_khr_fp64
#error "Double precision not supported on device."
#endif
#pragma OPENCL EXTENSION cl_khr_fp64: enable
{% endif %}
{% for c in consts -%}
  __constant {{ c.type }} {{ c.devname() }} = {{ c.hashdefine() }};
{% endfor %}

#include "{{ name }}_kernel.h"

