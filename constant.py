class Constant:
  def __init__(self, name=None, description=None, type='double', dim=1):
    if not name:
      raise Exception, "New Constant requires name"
    self.__name = name
    self.description = description
    self.type = type
    self.dim = dim

  def __repr__(self):
    return self.__name

  def name(self, pre='', suf=''):
    return pre + self.__name + suf

  def decl(self, pre='', suf='', include_dim=True):
    name = self.name(pre, suf)
    if self.dim > 1 and include_dim:
      return '%s %s[%d]' % (self.type, name, self.dim)
    else:
      return '%s %s' % (self.type, name)

  def devname(self):
    return 'D_' + self.__name.upper()

  def hashdefine(self):
    return 'CONSTANT_' + self.__name.upper()

  def sizeof(self):
    if self.dim == 1:
      return "sizeof(%s)" % self.type
    else:
      return "%d*sizeof(%s)" % (self.dim, self.type)
