import numpy as np
import sys

SEED = int(sys.argv[1])
DEFAULT_RNG = np.random.default_rng(SEED)

NUM_JOBS = int(sys.argv[2])

class Distribution:
  class_index = 0
  def __init__(self, distribution, params, rng=DEFAULT_RNG, qty=1):
    self.index_start = Distribution.class_index
    Distribution.class_index += qty
    self.index_end = Distribution.class_index
    self.distribution = distribution
    self.params = params
    self.rng = rng
  @property
  def index(self):
    return self.rng.integers(self.index_start, self.index_end)
  def set_rng(self, rng):
    self.rng = rng
  def __call__(self, *params, is_round=True, is_non_negative=True):
    if self.distribution == 'constant':
      result = self.params[0]
    else:
      result = getattr(self.rng, self.distribution)(*self.params)
    if is_non_negative: result = max(0, result)
    if is_round: result = round(result)
    return max(result, 1)

class Expo(Distribution):
  def __init__(self, *params, **kwargs): super().__init__('exponential', params, **kwargs)
class Normal(Distribution):
  def __init__(self, *params, **kwargs): super().__init__('normal', params, **kwargs)
class Poisson(Distribution):
  def __init__(self, *params, **kwargs): super().__init__('poisson', params, **kwargs)
class Constant(Distribution):
  def __init__(self, *params, **kwargs): super().__init__('constant', params, **kwargs)
  
steps = {
    'registration': Expo(5.5, qty=1),     # nurse
    'x_ray': Expo(12, qty=1),             # x-ray tech
    'consultation': Normal(15, 8, qty=1), # doctor
    'ct_scan': Normal(29, 14, qty=1),     # ct tech
    'dispensary': Normal(10, 8, qty=1),   # pharmacy
    # 'discharge': Normal(3, qty=1),        # nurse
}

patterns = [
    ['registration', 'x_ray', 'consultation', 'ct_scan', 'dispensary'],
    ['registration', 'x_ray', 'ct_scan', 'consultation', 'dispensary'],
    ['registration', 'consultation', 'x_ray', 'ct_scan', 'dispensary'],
    ['registration', 'consultation', 'ct_scan', 'x_ray', 'dispensary'],
    ['registration', 'ct_scan', 'x_ray', 'consultation', 'dispensary'],
    ['registration', 'ct_scan', 'consultation', 'x_ray', 'dispensary'],
]

f = lambda dist: (dist.index, dist())

machine2label = [None] * Distribution.class_index
for label, dist in steps.items():
    for i in range(dist.index_start, dist.index_end):
        machine2label[i] = label


try: instance_filename = sys.argv[3]
except: instance_filename = 'covid19'

with open(f'JSS/instances/{instance_filename}', 'w') as file:
    file.write(f'{NUM_JOBS} {Distribution.class_index}')
    file.write('\n')
    for _ in range(NUM_JOBS):
        pattern = DEFAULT_RNG.choice(patterns)
        file.write(' '.join(map(str,[x for step in pattern for x in f(steps[step])])))
        file.write('\n')

print(machine2label)