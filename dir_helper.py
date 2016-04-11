import os

def ensure(path):
  dname = os.path.dirname(path)

  if not os.path.exists(dname):
    os.makedirs(dname)
