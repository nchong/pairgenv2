#!/usr/bin/env python

from constant import *
from parameter import * 
import getopt, sys
import yaml
from jinja2 import Environment, PackageLoader

def flatten(l):
  return [item for sublist in l for item in sublist]

def usage():
  print "%s <params.yml>" % sys.argv[0]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    usage()
    exit(1)
  f = open(sys.argv[1], 'r')
  yaml_input = yaml.load(f)

  # top-level name and description of pairstyle
  name = yaml_input['name'].lower()
  desc = yaml_input['description']

  params = []
  for p in yaml_input['parameters']:
    if p.has_key('set'):
      if p['set'] == 'P':
        p['set'] = Set.P
      elif p['set'] == 'N':
        p['set'] = Set.N
      else:
        print "unrecognized set [%s]" % (p['set'])
    if p.has_key('access'):
      if p['access'] == 'RO':
        p['access'] = Access.RO
      elif p['access'] == 'RW':
        p['access'] = Access.RW
      elif p['access'] == 'SUM':
        p['access'] = Access.SUM
      else:
        print "unrecognized access [%s]" % (p['access'])
    p['name'].lower()
    params.append(Parameter(**p))

  consts = []
  for c in yaml_input['constants']:
    c['name'].lower()
    consts.append(Constant(**c))

  cl_khr_fp64 = \
    len([ p for p in params if p.type == 'double' ]) > 0 or \
    len([ c for c in consts if c.type == 'double' ]) > 0

  # process all templates in templates/ directory
  env = Environment(loader=PackageLoader('pairgen', 'templates'))
  for t in env.list_templates(filter_func=(lambda x: x[0] != '.')):
    output = name + '_' + t
    template = env.get_template(t)
    template.stream(
      cl_khr_fp64=cl_khr_fp64,
      name=name,
      headername=name.upper(),
      classname=name.capitalize(),
      params=params,
      consts=consts).dump(output)
   #print t, "->", output
