#ifndef {{ headername }}_CALC_H
#define {{ headername }}_CALC_H

#include <string>
using namespace std;

string human_readable(int nbytes);
string tostring(const char *l, int nbytes);
void print_sizes(int nparticles, int nneighbors);
int stoi(const char *s);

#endif

