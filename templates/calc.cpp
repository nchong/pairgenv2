/*
 * Simple calculator to estimate the dataset size of the pair style.
 */

#include "{{ name }}_calc.h"
#include <iomanip>
#include <iostream>
#include <sstream>

#define COLWIDTH 16

using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "Usage: ";
    cout << argv[0] << " <nparticles> <nneighbors>" << endl;
    return 1;
  }
  int nparticles = stoi(argv[1]);
  int nneighbors = stoi(argv[2]);

  print_sizes(nparticles, nneighbors);
  return 0;
}

string human_readable(int nbytes) {
  stringstream ss;
  if      (nbytes >= (0x1<<30)) ss << setprecision(4) << ((double)nbytes / (0x1<<30)) << "GB";
  else if (nbytes >= (0x1<<20)) ss << setprecision(4) << ((double)nbytes / (0x1<<20)) << "MB";
  else if (nbytes >= (0x1<<10)) ss << setprecision(4) << ((double)nbytes / (0x1<<10)) << "KB";
  return ss.str();
}

string tostring(const char *l, int nbytes) {
  stringstream ss;
  ss << setw(COLWIDTH) << l << setw(COLWIDTH) << nbytes << "B";
  string s = human_readable(nbytes);
  if (!s.empty()) {
    ss << setw(COLWIDTH) << s;
  }
  return ss.str();
}

void print_sizes(int nparticles, int nneighbors) {
  string line = " " + string(3*COLWIDTH, '-');
  cout << line << endl;
  cout << " {{ name }} dataset" << endl;
  cout << line << endl;
  cout << setw(COLWIDTH) << "nparticles" << setw(COLWIDTH) << nparticles << endl;
  cout << setw(COLWIDTH) << "nneighbors" << setw(COLWIDTH) << nneighbors << endl;
  cout << line << endl;

  long total = 0;
  // per-particle datasets
  {% for p in params if p.is_type('P', 'RO') %}
  {{- "// - read-only" if loop.first }}
  long {{ p.name(suf='_size') }} = nparticles * {{ p.dim }} * sizeof({{ p.type }});
  total += {{ p.name(suf='_size') }};
  cout << tostring("{{ p.name() }}", {{ p.name(suf='_size')}}) << endl;
  {%- endfor %}
  {% for p in params if p.is_type('P', 'RW') %}
  {{- "// - read-write" if loop.first }}
  long {{ p.name(suf='_size') }} = nparticles * {{ p.dim }} * sizeof({{ p.type }});
  total += {{ p.name(suf='_size') }};
  cout << tostring("{{ p.name() }}", {{ p.name(suf='_size')}}) << endl;
  {%- endfor %}
  {% for p in params if p.is_type('P', 'SUM') %}
  {{- "// - sum" if loop.first }}
  long {{ p.name(suf='_size') }} = nparticles * {{ p.dim }} * sizeof({{ p.type }});
  total += {{ p.name(suf='_size') }};
  cout << tostring("{{ p.name() }}", {{ p.name(suf='_size')}}) << endl;
  {%- endfor %}

  // per-neighbor datasets
  // these are lower-bounds because a neighbor list contains padding
  {% for p in params if p.is_type('N', 'RO') -%}
  {{- "// - read-only" if loop.first }}
  long {{ p.name(suf='_size') }} = nneighbors * {{ p.dim }} * sizeof({{ p.type }});
  total += {{ p.name(suf='_size') }};
  cout << tostring("{{ p.name() }}", {{ p.name(suf='_size')}}) << endl;
  {%- endfor %}
  {% for p in params if p.is_type('N', 'RW') %}
  {{- "// - read-write" if loop.first }}
  long {{ p.name(suf='_size') }} = nneighbors * {{ p.dim }} * sizeof({{ p.type }});
  total += {{ p.name(suf='_size') }};
  cout << tostring("{{ p.name() }}", {{ p.name(suf='_size')}}) << endl;
  {%- endfor %}

  cout << line << endl;
  cout << tostring("total", total) << endl;
  cout << line << endl;
}

int stoi(const char *s) {
  int x;
  istringstream iss(s);
  iss >> x;
  return x;
}

