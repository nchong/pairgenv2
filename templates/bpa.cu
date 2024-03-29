{% extends 'bases/bpa.template' %}
{% block extraheaders %}
#include "{{name}}_pair_kernel.cu"

// dynamically allocated at launch time
// see Section B2.3 in NVIDIA CUDA Programming Guide (v4.0)
extern __shared__ char array[];
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

{% block sharedmem %}
  {% set offset = "0" %}
  {% for p in params if p.is_type('P', 'SUM') %}
  {{- "// per-particle SUM data" if loop.first }}
  {{ p.type }} {{ p.name(pre='*l_', suf='_block') }} = ({{ p.type }} *)&array[{{ offset }}];
  {%- set offset = offset + " + (blockDim.x*%d*sizeof(%s))" % (p.dim, p.type) %}
  {%- else -%}
  {% endfor %}
{% endblock %}

{% block kidx %}
  int lid = threadIdx.x;
  int idx = blockIdx.x + (blockIdx.y * gridDim.x);
  int block_size = blockDim.x;
{% endblock %}

{% block smem_qualifier %} __shared__ {% endblock %}

{% block barrier %} __syncthreads(); {% endblock %}
