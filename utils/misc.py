from collections import namedtuple
import numpy as np

def pprinttable(rows):
  if len(rows) > 1:
    headers = rows[0]._fields
    lens = []
    for i in range(len(rows[0])):
      lens.append(len(max([x[i] for x in rows] + [headers[i]],key=lambda x:len(str(x)))))
    formats = []
    hformats = []
    for i in range(len(rows[0])):
      if isinstance(rows[0][i], int):
        formats.append("%%%dd" % lens[i])
      else:
        formats.append("%%-%ds" % lens[i])
      hformats.append("%%-%ds" % lens[i])
    pattern = " | ".join(formats)
    hpattern = " | ".join(hformats)
    separator = "-+-".join(['-' * n for n in lens])
    print(hpattern % tuple(headers))
    print(separator)
    _u = lambda t: t.decode('UTF-8', 'replace') if isinstance(t, bytes) else t
    for line in rows:
        print(pattern % tuple(_u(t) for t in line))
  elif len(rows) == 1:
    row = rows[0]
    hwidth = len(max(row._fields,key=lambda x: len(x)))
    for i in range(len(row)):
      print("%*s = %s" % (hwidth,row._fields[i],row[i]))

def display_table(iterable, header):
  Row = namedtuple('Row', header)
  pprinttable([Row(*row) for row in iterable])

def normalize_prob(freq_dict):
  keys, vals = zip(freq_dict.items())
  probs = np.asarray(vals, dtype=np.float64)
  probs /= probs.sum()
  return dict(zip(keys, probs))

def get_distribution_function(rng, distribution, params):
  if distribution == 'constant':
    return params[0]
  if distribution == 'poisson':
    return rng.poisson(*params)
  if distribution == 'expo':
    return rng.expo(*params)
  if distribution == 'normal':
    return rng.normal(*params)
  raise Exception(f"Distribution `{distribution}` is not recognized")