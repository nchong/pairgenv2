{%- macro memcpy_args(p) -%}
  {{- p.devname() -}}, 
  {{- p.sizeof() -}}, 
  {{- p.name(pre='h_') -}}
{%- endmacro -%}
#include "{{ name }}_wrapper.h"
#include "{{ name }}_clneighlist.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <iostream>

using namespace std;

#include "posix_timer.h"
#include <vector>
static SimpleTimer m0; static vector<double> m0_raw;
static SimpleTimer k0; static vector<double> k0_raw;
static SimpleTimer m1; static vector<double> m1_raw;
double get_m0() { return m0.total_time(); }
double get_k0() { return k0.total_time(); }
double get_m1() { return m1.total_time(); }
vector<double> &get_m0_raw() { return m0_raw; }
vector<double> &get_k0_raw() { return k0_raw; }
vector<double> &get_m1_raw() { return m1_raw; }

{{ classname }}Wrapper::{{ classname }}Wrapper(
    CLWrapper &clw, size_t wx,
    int N, int maxpage, int pgsize,
    {% for c in consts -%}
      {{ c.decl(pre='h_', include_dim=False) }},
    {% endfor -%}
    {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {% endfor -%}
  ) :
  clw(clw), N(N), wx(wx), tpa_gx(wx * ((N/wx)+1)), bpa_gx(wx * N),
  d_nl(new {{ classname }}CLNeighList(clw, wx, N, maxpage, pgsize))
{
    {% for p in params if p.is_type('P', '-') -%}
    {{ p.devname() }} = clw.dev_malloc({{ p.sizeof() }});
    {% endfor %}
    {% for p in params if p.is_type('P', 'RO') and not p.reload -%}
      clw.memcpy_to_dev({{ memcpy_args(p) }});
    {% endfor %}
    stringstream extra_flags;
    extra_flags << " -I {{ cwd }}";
    extra_flags << " -D BLOCK_SIZE=" << wx;
    {% for c in consts -%}
    extra_flags << " -D {{ c.hashdefine() }}=" << {{ c.name(pre='h_')}};
    {% endfor %}
#ifdef STAGE_PARTICLE_I_DATA
    extra_flags << " -D STAGE_PARTICLE_I_DATA";
#ifdef RANGECHECK
    extra_flags << " -D RANGECHECK";
#endif
#endif
#ifdef STAGE_NEIGHBOR_DATA
    extra_flags << " -D STAGE_NEIGHBOR_DATA";
#endif
#if DEBUG
    cerr << "[DEBUG] Compiling with extra_flags = [" << extra_flags.str() << "]" << endl;
    cerr << "[DEBUG] Kernel TpA parameters gx=" << tpa_gx << " wx=" << wx << endl;
    cerr << "[DEBUG] Kernel BpA parameters gx=" << bpa_gx << " wx=" << wx << endl;
#endif
    clw.create_all_kernels(clw.compile("{{name}}_tpa.cl", extra_flags.str()));
    tpa = clw.kernel_of_name("{{name}}_tpa");
    clw.create_all_kernels(clw.compile("{{name}}_bpa.cl", extra_flags.str()));
    bpa = clw.kernel_of_name("{{name}}_bpa");
}

{{ classname }}Wrapper::~{{ classname }}Wrapper() {
  {% for p in params if not p.is_type('N', '-') -%}
    clw.dev_free({{ p.devname() }});
  {% endfor %}
  delete(d_nl);
}

void {{ classname }}Wrapper::refill_neighlist(
  int *h_numneigh,
  int **h_firstneigh,
  int **h_pages,
  int maxpage,
  {% for p in params if p.is_type('N', '-') -%}
    {{ p.pages() }}{{ ',' if not loop.last }}
  {% endfor -%}
) {
  d_nl->reload(h_numneigh, h_firstneigh, h_pages, maxpage,
      {%- for p in params if p.is_type('N', '-') -%}
        {{ p.name(pre='h_',suf='pages') }}{{ ', ' if not loop.last }}
      {%- endfor -%}
  );
}

void {{ classname }}Wrapper::run(
  kernel_decomposition kernel,
  {% for p in params -%}
    {%- if p.is_type('P', 'RO') and p.reload -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {%- elif p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
      {{ p.decl(pre='*h_', include_dim=False) }}{{ ',' if not loop.last }}
    {%- elif p.is_type('N', 'RO') and p.reload -%}
      {{ p.pages() }}{{ ',' if not loop.last }}
    {%- elif p.is_type('N', 'RW') -%}
      {{ p.pages() }}{{ ',' if not loop.last }}
    {%- else -%}
      // {{ p.name() }} is not reloaded
    {%- endif %}
  {% endfor -%}
) {
  m0.start();
  {% for p in params if p.is_type('P', '-') and p.reload -%}
    clw.memcpy_to_dev({{ memcpy_args(p) }});
  {% endfor -%}
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    clw.memcpy_to_dev({{ memcpy_args(p) }});
  {% endfor %}
  m0_raw.push_back(m0.stop_and_add_to_total());

  k0.start();
  if (kernel == TPA) {
    clw.kernel_arg(tpa,
      N,
      {% for p in params if p.is_type('P', 'RO') -%}
      {{ p.devname() }},
      {% endfor -%}
      d_nl->d_numneigh, d_nl->d_offset, d_nl->d_neighidx,
      {% for p in params if not p.is_type('P', 'RO') -%}
        {%- if p.is_type('N', '-') -%}
      d_nl->{{ p.devname() }}{{ ', ' if not loop.last }}
        {%- else -%}
      {{ p.devname() }}{{ ', ' if not loop.last }}
        {%- endif -%}
      {% endfor -%}
    );
    clw.run_kernel(tpa, /*dim=*/1, &tpa_gx, &wx);
  } else if (kernel == BPA) {
    {% for p in params if p.is_type('P', 'SUM') %}
    size_t {{ p.name(pre='l_', suf='_block_size') }} = wx*{{ p.dim }}*sizeof({{ p.type }});
    {%- endfor %}
    clw.kernel_arg(bpa,
      N,
      {% for p in params if p.is_type('P', 'RO') -%}
      {{ p.devname() }},
      {% endfor -%}
      d_nl->d_numneigh, d_nl->d_offset, d_nl->d_neighidx,
      {%- for p in params if not p.is_type('P', 'RO') %}
        {%- if p.is_type('N', '-') %}
      d_nl->{{ p.devname() }}{{ ', ' if not loop.last }}
        {%- elif p.is_type('P', 'SUM') %}
      {{ p.devname() }},
      {{ p.name(pre='l_', suf='_block_size') }}{{ ', ' if not loop.last }}
        {%- else %}
      {{ p.devname() }}{{ ', ' if not loop.last }}
        {%- endif -%}
      {% endfor -%}
    );
    clw.run_kernel(bpa, /*dim=*/1, &bpa_gx, &wx);
  }
  clw.flush_command_queue();
  k0_raw.push_back(k0.stop_and_add_to_total());

  m1.start();
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    clw.memcpy_from_dev({{ memcpy_args(p) }});
  {% endfor %}

  {% for p in params if p.is_type('N', 'RW') -%}
  if ({{ p.name(pre='h_',suf='pages') }} != NULL) {
    d_nl->unload_{{ p.name() }}({{ p.name(pre='h_',suf='pages') }});
  }
  {% endfor %}
  m1_raw.push_back(m1.stop_and_add_to_total());
}

