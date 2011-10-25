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

#ifndef {{ headername }}_BPA_H
#define {{ headername }}_BPA_H
#include "{{name}}_pair_kernel.cu"

/*
 * CUDA block-per-particle decomposition.
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
 * We assign one block per particle i.
 * Each thread within a block is assigned one of the numneigh[i] neighbors of i.
 * NB: This kernel does not update any neighbor particles.
 *     Therefore the neighbor list must contain symmetric duplicates.
 */

// dynamically allocated at launch time
// see Section 4.2.2.3 in NVIDIA CUDA Programming Guide (v2.0)
extern __shared__ char array[];

__global__ void {{name}}_bpa(
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
  {% set offset = "0" %}
  {% for p in params if p.is_type('P', 'SUM') %}
  {{- "// block-level array for per-particle SUM data" if loop.first }}
  {{ p.type }} {{ p.name(pre='*l_', suf='_block') }} = ({{ p.type }} *)&array[{{ offset }}];
  {%- set offset = offset + " + (blockDim.x*%d*sizeof(%s))" % (p.dim, p.type) %}
  {%- else -%}
  {% endfor %}

  // register copies of particle and neighbor data
  {%- for p in params if not p.is_type('P', 'SUM') -%}
    {%- for n in p.tagged_name() %}
  {{ p.type }} {{ n }}{{ "[%d]" % p.dim if p.dim == 3 }};
    {%- endfor -%}
  {% endfor %}

  int idx = blockIdx.x;
  int nneigh = numneigh[idx];
  int block_size = blockDim.x;

  for (int jj=threadIdx.x; jj<nneigh; jj+=block_size) {
    {% for p in params if p.is_type('P', 'SUM') %}
    {{- "// block-level per-particle SUM data" if loop.first -}}
      {%- if p.dim > 1 %}
        {%- for k in range(p.dim) %}
    {{ p.name(pre='l_', suf='_block') }}[(jj*{{ p.dim }})+{{ k }}] = 0;
        {%- endfor %}
      {%- else %}
    {{ p.name(pre='l_', suf='_block') }}[jj] = 0;
      {%- endif %}
    {%- else -%}
    {%- endfor %}
  }

  {% for p in params if p.is_type('P', 'SUM') %}
  {{- "// thread-local per-particle SUM data" if loop.first -}}
    {%- if p.dim > 1 %}
  {{ p.type }} {{ p.name(suf='i_delta') }}[{{ p.dim }}] = {{ p.additive_id() }};
    {%- else %}
  {{ p.type }} {{ p.name(suf='i_delta') }} = {{ p.additive_id() }};
    {%- endif %}
  {%- else -%}
  {%- endfor %}

  if (idx < N && nneigh > 0) {
    for (int jj=threadIdx.x; jj<nneigh; jj+=block_size) {
      {% for p in params if p.is_type('P', 'RO') %}
      {{- "// per-particle RO data" if loop.first -}}
        {{- assign(p, 'i', 'idx')|indent(2) -}}
      {%- else -%}
      {%- endfor %}
      {#- TODO: DEAL WITH P,RW DATA #}

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

      // writeback per-neighbor RW data
      {%- for p in params if p.is_type('N', 'RW') -%}
        {{- assigninv(p,'','nidx')|indent(2) -}}
      {%- endfor %}
    }
  }

  {% for p in params if p.is_type('P', 'SUM') %}
  {{- "// write per-particle SUM data into block-shared arrays" if loop.first -}}
    {%- if p.dim > 1 %}
      {%- for k in range(p.dim) %}
  {{ p.name(pre='l_', suf='_block') }}[(threadIdx.x*{{ p.dim }})+{{ k }}] = {{ p.name(suf='i_delta') }}[{{ k }}];
      {%- endfor %}
    {%- else %}
  {{ p.name(pre='l_', suf='_block') }}[threadIdx.x] = {{ p.name(suf='i_delta') }};
    {%- endif %}
  {%- else -%}
  {%- endfor %}

  __syncthreads();

  // local reduce
  if (idx < N && nneigh > 0 && threadIdx.x == 0) {
    for (int i=1; i<nneigh; i++) {
      {%- for p in params if p.is_type('P', 'SUM') %}
        {%- if p.dim > 1 %}
          {%- for k in range(p.dim) %}
      {{ p.name(suf='i_delta') }}[{{ k }}] += {{ p.name(pre='l_', suf='_block') }}[(i*{{ p.dim }})+{{ k }}];
          {%- endfor %}
        {%- else %}
      {{ p.name(suf='i_delta') }} += {{ p.name(pre='l_', suf='_block') }}[i];
        {%- endif %}
      {%- else -%}
      {%- endfor %}
    }

    {%- for p in params if p.is_type('P', 'SUM') %}
      {%- if p.dim > 1 %}
        {%- for k in range(p.dim) %}
    {{ p.name(pre='d_') }}[idx*{{ p.dim }}+{{ k }}] += {{ p.name(suf='i_delta') }}[{{ k }}];
        {%- endfor %}
      {%- else %}
    {{ p.name(pre='d_') }}[idx] += {{ p.name(suf='i_delta') }};
      {%- endif %}
    {%- else -%}
    {%- endfor %}
  }

}

#endif
