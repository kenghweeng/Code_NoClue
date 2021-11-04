from env import EDPSEnv
from pprint import pprint
import numpy as np

basic_config = {
    "resources": {0: 1}, 
    "acuities": 1, 
    "prob_acuities": [1], 
    "weighted_wait": {0: 1},
    "orders": [[0]],
    "spawn": 5, 
    "treatment_times": {0: {0: 1}}, 
    "max_time": 60, 
    "set_seed": 0
}

class Distribution:
    def __init__(self, distribution, params):
        self.distribution = distribution
        self.params = params
        self.rng = None
    def set_rng(self, rng):
        self.rng = rng
    def __call__(self, is_round=True):
        result = max(0, getattr(self.rng, self.distribution)(*self.params))
        return round(result) if is_round else result

class Expo(Distribution):
    def __init__(self, *params): super().__init__('exponential', params)
class Normal(Distribution):
    def __init__(self, *params): super().__init__('normal', params)
class Constant(Distribution):
    def __init__(self, *params): super().__init__('constant', params)
    def __call__(self): return self.params[0]

# Improving Emergency Department Efficiency by Patient Scheduling Using Deep Reinforcement Learning
# https://doi.org/10.3390/healthcare8020077
paper_config = {
    "resources": {0: 3, 1: 5, 2: 1, 3: 1}, # pg 10, para 2
    "acuities": 5,                         # pg 9, table 4, Acuity Level
    "prob_acuities": [.1,.3,.4,.1,.1],     # pg 9, table 4, Ratio
    "weighted_wait": [30,15,1,1,1],        # pg 9, table 4, Weighted Waiting Time
    "orders": [[0,1,2,7],[0,1,2,4,7],[0,1,2,3,4,7],[0,1,2,5,7],[0,1,2,3,4,5,7],[0,1,2,3,7],[0,1,2,4,3,7],[0,1,2,6,7],[0,1,2,6,4,5,8],[0,1,2,3,8],[0,1,2,4,3,8]],
    "spawn": 5, # poisson 7, 8, 9, 10
    "treatment2resource": {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
        4: 2,
        5: 1,
        6: 3,
        7: 1,
        8: 1,
    },
    "treatment_times": { # pg 8, table 2
        # resource: treatment
        0: {0: Expo(7), 2: Normal(14, 6)},
        1: {1: Expo(5.5), 3: Normal(35, 15), 5: Normal(15, 8), 7: Constant(30), 8: Expo(3)},
        2: {4: Expo(12)},
        3: {6: Normal(29, 14)},
    },
    "max_time": 60, # 60*24*14
    "set_seed": 0
}

simple_config = {
    "resources": {0: 3, 1: 5, 2: 1}, # 3 docs, 5 nurses, 1 xray (no more than 6, no more than 4)
    "acuities": 5,                         # 5 severities
    "prob_acuities": [.1,.3,.4,.1,.1],     # acuity probability frequencies
    "weighted_wait": [30,15,1,1,1],        # Weighted Waiting Time
    "orders": [[1,0],[1,2,0]],             # triage>consult; triage>xray>consult
    "spawn": 5, # poisson 7, 8, 9, 10
    "treatment2resource": {
        0: 0,
        1: 1,
        2: 2,
    },
    "treatment_times": { # pg 8, table 2
        # resource: treatment
        0: {0: Expo(7)},
        1: {1: Expo(5.5)},
        2: {2: Expo(12)},
    },
    "max_time": 60, # 60*24*14
    "set_seed": 0
}

basic_edps = EDPSEnv(simple_config)
obs = basic_edps.reset()
initial_debug = basic_edps.debug()

obs = basic_edps._get_state()

done = False
while not done:
    print(obs)
    action = obs.argmax(axis=1)
    obs, reward, done, debug = basic_edps.step(action)
    pprint(debug)
