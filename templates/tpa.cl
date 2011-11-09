{% from 'macros/parameter.jinja' import localassign, assign, assigninv %}
#ifndef {{ headername }}_TPA_H
#define {{ headername }}_TPA_H
{% if cl_khr_fp64 %}
#ifndef cl_khr_fp64
#error "Double precision not supported on device."
#endif
#pragma OPENCL EXTENSION cl_khr_fp64: enable
{% endif %}
#include "{{name}}_pair_kernel.cl"

#ifndef BLOCK_SIZE
#error You need to #define BLOCK_SIZE
#endif

/*
 * OpenCL thread-per-particle decomposition.
 *
 * Given N particles,
 *  d_<var>[i]
 *  is the particle data for particle i
 *
 *  numneigh[i] 
 *  is the number of neighbors for particle i
 *
 *  neighidx[offset[i] + jj]
 *  is the index j of the jj-th neighbor to particle i
 *
 * We assign one thread per particle i.
 * Each thread loops over the numneigh[i] neighbors of i.
 * NB: This kernel does not update any neighbor particles.
 *     Therefore the neighbor list must contain symmetric duplicates.
 */
__kernel void {{name}}_tpa(
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
  ) {
  // local, block and global identifiers
  int lid = get_local_id(0);
  int bid = get_group_id(0);
  int idx = get_global_id(0);

  // load particle(i) data into shared memory
  {%- for p in params if p.is_type('P', 'RO') %}
  __local {{ p.type }} {{ p.name(pre='local_') }}[BLOCK_SIZE*{{ p.dim }}];
  {%- endfor %}
#ifdef RANGECHECK
  {%- for p in params if p.is_type('P', 'RO') %}
  {%- if p.dim > 1 %}
    {% for k in range(p.dim) %}
  {{ p.name(pre='local_') }}[(BLOCK_SIZE*{{ k }})+lid] = {{ p.devname() }}[(bid*{{ p.dim }})+(BLOCK_SIZE*{{ k }})+lid];
    {%- endfor -%}
  {%- else %}
  {{ p.name(pre='local_') }}[lid] = {{ p.devname() }}[idx];
  {%- endif %}
  {%- endfor %}
#else //use modulo wrapping
  {%- for p in params if p.is_type('P', 'RO') %}
  {%- if p.dim > 1 %}
    {%- for k in range(p.dim) %}
  {{ p.name(pre='local_') }}[(BLOCK_SIZE*{{ k }})+lid] = {{ p.devname() }}[((bid*{{ p.dim }})+(BLOCK_SIZE*{{ k }})+lid)%N];
    {%- endfor -%}
  {%- else %}
  {{ p.name(pre='local_') }}[lid] = {{ p.devname() }}[idx%N];
  {%- endif %}
  {%- endfor %}
#endif
  // required because we stride across per-particle data of dim>1 in block steps to ensure coalescing
  // ie, this thread does not consume what it loads (except for the first load)
  mem_fence(CLK_LOCAL_MEM_FENCE);

  // register copies of particle and neighbor data
  {%- for p in params if not p.is_type('P', 'SUM') -%}
    {%- for n in p.tagged_name() %}
  {{ p.type }} {{ n }}{{ "[%d]" % p.dim if p.dim == 3 }};
    {%- endfor -%}
  {% endfor %}
  if (idx < N && numneigh[idx] > 0) {
    // load particle i data
    {% for p in params if p.is_type('P', 'RO') %}
    {{- "// per-particle RO data" if loop.first -}}
      {{- localassign(p, 'i', 'lid') -}}
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
    int myoffset = offset[idx];
    for (int jj=0; jj<numneigh[idx]; jj++) {
      // load particle j data
      int nidx = myoffset + jj;
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

