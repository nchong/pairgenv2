{% macro assign(p,suf,idx) -%}
  {% if p.dim > 1 -%}
    {% for k in range(p.dim) %}
    {{ p.name(suf=suf) }}[{{ k }}] = {{ p.devname() }}[({{ idx }}*{{ p.dim }})+{{k}}];
    {%- endfor -%} 
  {%- else %}
    {{ p.name(suf=suf) }} = {{ p.devname() }}[{{ idx }}];
  {%- endif -%}
{% endmacro %}
{% macro assigninv(p,suf,idx,sum=false) -%}
  {% if p.dim > 1 -%}
    {% for k in range(p.dim) %}
    {{ p.devname() }}[({{ idx }}*{{ p.dim }})+{{k}}] {{'+' if sum}}= {{ p.name(suf=suf) }}[{{ k }}];
    {%- endfor -%} 
  {%- else %}
    {{ p.devname() }}[{{ idx }}] {{'+' if sum}}= {{ p.name(suf=suf) }};
  {%- endif -%}
{% endmacro %}

#ifndef {{ headername }}_TPA_H
#define {{ headername }}_TPA_H
#include "{{name}}_pair_kernel.cu"

/*
 * CUDA thread-per-particle decomposition.
 *
 * Given N particles,
 *  d_<var>[i]
 *  is the particle data for particle i
 *
 *  numneigh[i] 
 *  is the number of neighbors for particle i
 *
 *  neighidx[(pageidx[i] * pgsize) + offset[i] + jj]
 *  is the index j of the jj-th neighbor to particle i
 *
 * We assign one thread per particle i.
 * Each thread loops over the numneigh[i] neighbors of i.
 * NB: This kernel does not update any neighbor particles.
 *     Therefore the neighbor list must contain symmetric duplicates.
 */
__global__ void {{name}}_tpa(
  int N, // number of particles
  {% for p in params if p.is_type('P', 'RO') -%}
  {{ p.type }} {{ p.devname(pre='*') }},
  {% endfor -%}
  int *numneigh,
  int *pageidx,
  int *offset,
  int pgsize,
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
  ) {
  // register copies of particle and neighbor data
  {%- for p in params if not p.is_type('P', 'SUM') -%}
    {%- for n in p.tagged_name() %}
  {{ p.type }} {{ n }}{{ "[%d]" % p.dim if p.dim == 3 }};
    {%- endfor -%}
  {% endfor %}
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N && numneigh[idx] > 0) {
    // load particle i data
    {% for p in params if p.is_type('P', 'RO') %}
    {{- "// per-particle RO data" if loop.first -}}
      {{- assign(p, 'i', 'idx') -}}
    {%- else -%}
    {%- endfor %}
{#- TODO: DEAL WITH P,RW DATA -#}
{#-  {% for p in params if p.is_type('P', 'RW') %}
    {{- "// per-particle RW data" if loop.first -}}
      {{- assign(p, 'i', 'idx') -}}
    {%- else -%}
    {%- endfor %}
#}
    {% for p in params if p.is_type('P', 'SUM') %}
    {{- "// per-particle SUM data" if loop.first -}}
      {%- if p.dim > 1 %}
    {{ p.type }} {{ p.name(suf='i_delta') }}[{{ p.dim }}] = {{ p.additive_id() }};
      {%- else %}
    {{ p.type }} {{ p.name(suf='i_delta') }} = {{ p.additive_id() }};
      {%- endif %}
    {%- else -%}
    {%- endfor %}

    // iterate over each neighbor of particle i
    for (int jj=0; jj<numneigh[idx]; jj++) {
      // load particle j data
      int mypage = pageidx[idx];
      int myoffset = offset[idx];
      int nidx = (mypage*pgsize) + myoffset + jj;
      int j = neighidx[nidx];
      {%- for p in params if p.is_type('P', 'RO') %}
      {{- assign(p, 'j', 'j')|indent(2) -}}
      {%- endfor %}
      {# not possible to load per-particle j data #}
      // load neighbor(i,j) data
      {%- for p in params if p.is_type('N', '-') %}
      {{- assign(p, '', 'nidx')|indent(2) -}}
      {%- endfor %}

      // do pairwise calculation
      {{ name }}_pair_kernel(
        {%- for p in params -%}
          {% set outer_loop = loop %}
          {% for n in p.tagged_name() -%}
            {% set comma = not (outer_loop.last and loop.last) %}
            {%- if p.dim == 1 -%}
      {{ '&' if p.is_type('-', 'RW') or p.is_type('-','SUM') }}{{ n }}{{ ', ' if comma }}
            {%- else -%}
      {{ n }}{{ ', ' if comma }}
            {%- endif -%}
          {%- endfor -%}
        {%- endfor -%}
      );
      {#- TODO: DEAL WITH P,RW DATA -#}
      // writeback per-neighbor RW data
      {%- for p in params if p.is_type('N', 'RW') -%}
        {{- assigninv(p,'','nidx')|indent(2) -}}
      {%- endfor %}
    }

    // writeback per-particle SUM data
    {%- for p in params if p.is_type('P', 'SUM') -%}
      {{- assigninv(p,'i_delta','idx',true) -}}
    {%- endfor %}
  }
}

#endif

