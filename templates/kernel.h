{%- macro comment(header, set, access) %}
 * {{ header }}
{%- for p in params if p.is_type(set, access) %}
 * {{"\t"}} {%-   for n in p.tagged_name() -%}
   {{ n }}{{ ', ' if not loop.last }}
   {%- endfor -%}
   {{- "\t\t(* %s *)" % p.description if p.description -}}
{%- else %}
 * {{"\t"}}--
{%- endfor -%}
{%- endmacro %}
{%- macro constant_comment(header) %}
 * {{ header }}
{%- for c in consts %}
 * {{"\t"}}
   {{- c.devname() }}
   {{- "\t\t(* %s *)" % c.description if c.description -}}
{%- else %}
 * {{"\t"}}--
{%- endfor -%}
{%- endmacro %}
/*
 * Pairwise interaction of particles i and j
 *
 {{- comment('Read-only per-particle:', 'P', 'RO')  -}}
 {{  comment('Read-write per-particle:', 'P', 'RW') -}}
 {{  comment('Sum-into per-particle:', 'P', 'SUM')   }}
 *
 {{- comment('Read-only per-neighbor:', 'N', 'RO')  -}}
 {{  comment('Read-write per-neighbor:', 'N', 'RW')  }}
 *
 {{- constant_comment('Constants:')                  }}
 */
#if defined(__CUDACC__)
  __device__
#else
  inline
#endif
void {{name}}_pair_kernel(
  {%- for p in params -%}
    {% set outer_loop = loop %}
    {% for n in p.tagged_name() -%}
      {% set comma = not (outer_loop.last and loop.last) %}
      {%- if p.dim == 1 -%}
   {{ p.type }} {{ '*' if p.is_type('-', 'RW') or p.is_type('-','SUM') }}{{ n }}{{ ', ' if comma }}
      {%- else -%}
   {{ p.type }} {{ n }}[{{ p.dim }}]{{ ', ' if comma }}
      {%- endif -%}
    {%- endfor -%}
  {%- endfor -%}
) {
  //fill me in
}

