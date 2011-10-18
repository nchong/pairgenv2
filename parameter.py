def enum(*sequential, **named):
  enums = dict(zip(sequential, range(len(sequential))), **named)
  return type('Enum', (), enums)

Set = enum('P', 'N')
Access = enum('RO', 'RW', 'SUM')

def string_of_set(s):
  map = { Set.P: 'P', Set.N: 'N' }
  return map[s]

def string_of_access(a):
  map = { Access.RO: 'RO', Access.RW: 'RW', Access.SUM: 'SUM' }
  return map[a]

class Parameter:
  def __init__(self, name=None, type='double', dim=3,
               set=Set.P, access=Access.RO, reload=False, identity=None,
               description=None):
    if not name:
      raise Exception, "New Parameter requires name"
    if set == Set.N and access == Access.SUM:
      raise Exception, 'Per-neighbor sum parameter is not supported'
    if set not in [ Set.N, Set.P ]:
      raise Exception, 'Unknown set [' + set + ']'
    if access not in [ Access.RO, Access.RW, Access.SUM ]:
      raise Exception, 'Unknown access [' + access + ']'
    if dim not in [ 1, 3]:
      raise Exception, 'Unknown dim [' + dim + ']'
    if type not in [ 'double', 'int' ]:
      raise Exception, 'Unknown type [' + type + ']'
    self.__name = name
    self.type = type
    self.dim = dim
    self.__set = set
    self.__access = access
    self.reload = reload
    self.__identity = identity
    self.description = description

  def __eq__(self, other):
    if self.__name      != other.__name: return False
    if self.type        != other.type: return False
    if self.dim         != other.dim: return False
    if self.__set       != other.__set: return False
    if self.__access    != other.__access: return False
    if self.reload      != other.reload: return False
    if self.__identity  != other.__identity: return False
    if self.description != other.description: return False
    return True

  def __repr__(self):
    try:
      return '%s(%s,%s)' % (self.__name, string_of_set(self.__set), string_of_access(self.__access))
    except KeyError:
      return self.__name

  # '-' is wildcard
  def is_type(self, set, access):
    set_map = {'P': Set.P,
               'N': Set.N,
               '-': self.__set}
    access_map = {'RO' : Access.RO,
                  'RW' : Access.RW,
                  'SUM': Access.SUM,
                  '-': self.__access}
    try:
      return (self.__set, self.__access) == (set_map[set], access_map[access])
    except KeyError:
      raise Exception, 'Unknown set [%s] or access[%s]' % (set, access)

  def name(self, pre='', suf=''):
    return pre + self.__name + suf

  def devname(self, pre='', suf=''):
    return self.name(pre=pre+'d_', suf=suf)

  def tagged_name(self):
    if self.is_type('P', 'RO'):
      return [self.name(suf='i'), 
              self.name(suf='j')]
    elif self.is_type('P', 'RW'):
      return [self.name(suf='i')]
    elif self.is_type('P', 'SUM'):
      return [self.name(suf='i_delta')]
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return [self.name()]

  def decl(self, pre='', suf='', include_dim=True):
    name = self.name(pre, suf)
    if self.dim > 1 and include_dim:
      return '%s %s[%d]' % (self.type, name, self.dim)
    else:
      return '%s %s' % (self.type, name)

  def tagged_decl(self):
    if self.is_type('P', 'RO'):
      return [self.decl(suf='i'), 
              self.decl(suf='j')]
    elif self.is_type('P', 'RW'):
      return [self.decl(suf='i')]
    elif self.is_type('P', 'SUM'):
      return [self.decl(suf='i_delta')]
    elif self.is_type('N', 'RO') or self.is_type('N', 'RW'):
      return [self.decl()]

  def additive_id(self):
    if self.__identity:
      return self.__identity 
    map = {'double' : '0.0f', 'int' : '0'}
    id = map[self.type]
    if self.dim > 1:
      return '{' + ', '.join([id]*self.dim) + '}'
    else:
      return id

  def sizeof(self):
    if self.__set == Set.P:
      return "N*%d*sizeof(%s)" % (self.dim, self.type)
    elif self.__set == Set.N:
      return "maxpage*pgsize*%d*sizeof(%s)" % (self.dim, self.type)
    else:
      raise Exception, "Unknown set"

  def printf_placeholder(self):
    map = { 'double':'%.16f', 'int':'%d' }
    try:
      return map[self.type]
    except KeyError:
      raise Exception, 'Unknown printf_placeholder(%s)' % (self.type)

  def pages(self):
    if self.__set == Set.P:
      raise Exception, "Not defined"
    elif self.__set == Set.N:
      return self.decl(pre='**h_', suf='pages', include_dim=False)
