{% extends 'bases/tpa.template' %}
{% block extraheaders %}
#include "{{name}}_pair_kernel.cu"
{% endblock %}

{% block kqualifier %} __global__ {% endblock %}

{% block kparameters %}
  int N, // number of particles
  {% for p in params if p.is_type('P', 'RO') -%}
  {{ p.type }} {{ p.devname(pre='*') }},
  {% endfor -%}
  int *numneigh,
  int *offset,
  int *neighidx
  {%- for p in params if not p.is_type('P', 'RO') -%}
  {{- ', ' if loop.first }}
  {%- if p.is_type('P', '-') %}
  {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- elif p.is_type('N', '-') %}
  {{ p.decl(pre='*d_', include_dim=False) }}{{ ', ' if not loop.last }}
  {%- else %}
    #error pairgen generation problem (this should be unreachable!)
  {%- endif %}
  {%- endfor -%}
{% endblock %}

{% block sharedmem %} {% endblock %}

{% block kidx %}
  int lid = threadIdx.x;
  int bid = blockIdx.x * blockDim.x;
  int idx = threadIdx.x + bid;
{% endblock %}

{% block smem_qualifier %} __shared__ {% endblock %}

{% block barrier %} __syncthreads(); {% endblock %}
