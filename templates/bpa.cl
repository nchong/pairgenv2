{% extends 'bases/bpa.template' %}
{% block extraheaders %}
#include "{{name}}_pair_kernel.cl"

{% if cl_khr_fp64 %}
#ifndef cl_khr_fp64
#error "Double precision not supported on device."
#endif
#pragma OPENCL EXTENSION cl_khr_fp64: enable
{% endif %}
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
  {%- if p.is_type('P', 'RW') %}
  __global {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- elif p.is_type('P', 'SUM') %}
  __global {{ p.decl(pre='*d_', include_dim=False) }},
  __local  {{ p.decl(pre='*l_', suf='_block', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- elif p.is_type('N', '-') %}
  __global {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- else %}
    #error pairgen generation problem (this should be unreachable!)
  {%- endif %}
  {%- endfor -%}
{% endblock %}

{% block sharedmem %} {# empty #} {% endblock %}

{% block kidx %}
  int lid = get_local_id(0);
  int idx = get_group_id(0);
  int block_size = get_local_size(0);
{% endblock %}

{% block smem_qualifier %} __local {% endblock %}

{% block barrier %} barrier(CLK_LOCAL_MEM_FENCE); {% endblock %}
