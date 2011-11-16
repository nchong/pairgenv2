{% extends 'bases/tpa.template' %}
{% block extraheaders %}
#include "{{name}}_pair_kernel.cl"

{% if cl_khr_fp64 %}
#ifndef cl_khr_fp64
#error "Double precision not supported on device."
#endif
#pragma OPENCL EXTENSION cl_khr_fp64: enable
{% endif %}

#ifndef BLOCK_SIZE
#error You need to #define BLOCK_SIZE
#endif
{% endblock %}

{% block kqualifier %} __kernel {% endblock %}

{% block kparameters %}
  int N, // number of particles
  {% for p in params if p.is_type('P', 'RO') -%}
  __global {{ p.type }} {{ p.devname(pre='*') }},
  {% endfor -%}
  __global int *numneigh,
  __global int *offset,
  __global int *neighidx
  {%- for p in params if not p.is_type('P', 'RO') -%}
  {{- ', ' if loop.first }}
  {%- if p.is_type('P', '-') %}
  __global {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- elif p.is_type('N', '-') %}
  __global {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- else %}
    #error pairgen generation problem (this should be unreachable!)
  {%- endif %}
  {%- endfor -%}
{% endblock %}

{% block sharedmem %}
  {%- for p in params if p.is_type('P', 'RO') %}
  __local {{ p.type }} {{ p.name(pre='local_') }}[BLOCK_SIZE*{{ p.dim }}];
  {%- endfor %}
{% endblock %}

{% block kidx %}
  int lid = get_local_id(0);
  int bid = get_group_id(0) * get_local_size(0);
  int idx = get_global_id(0);
{% endblock %}

{% block memfence %} mem_fence(CLK_LOCAL_MEM_FENCE); {% endblock %}
