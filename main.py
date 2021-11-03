from env import EDPSEnv
from pprint import pprint
import numpy as np

basic_config = {
    "resources": {0: 1}, 
    "acuities": 1, 
    "prob_acuities": [1], 
    "weighted_wait": {0: 1},
    "order": [0],
    "spawn": 5, 
    "treatment_times": {0: {0: 1}}, 
    "max_time": 60, 
    "set_seed": 0
}

# Improving Emergency Department Efficiency by Patient Scheduling Using Deep Reinforcement Learning
# https://doi.org/10.3390/healthcare8020077
paper_config = {
    "resources": {0: 1}, 
    "acuities": 5,                     # pg 9, table 4, Acuity Level
    "prob_acuities": [.1,.3,.4,.1,.1], # pg 9, table 4, Ratio
    "weighted_wait": [30,15,1,1,1],    # pg 9, table 4, Weighted Waiting Time
    "order": [0],
    "spawn": 5, 
    "treatment_times": {0: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}},
    "max_time": 60, 
    "set_seed": 0
}

basic_edps = EDPSEnv(paper_config)
obs = basic_edps.reset()
initial_debug = basic_edps.debug()

obs = basic_edps._get_state()

done = False
while not done:
    action = np.unravel_index(obs.argmax(), obs.shape)
    obs, reward, done, debug = basic_edps.step(action)
    pprint(debug)
