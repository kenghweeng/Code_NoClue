from env import EDPSEnv
from pprint import pprint

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

basic_edps = EDPSEnv(basic_config)
obs = basic_edps.reset()
initial_debug = basic_edps.debug()

done = False
while not done:
    obs, reward, done, debug = basic_edps.step((0,0))
    pprint(debug)
