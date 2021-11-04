from env import EDPSEnv
from pprint import pprint
import numpy as np
from utils.config import Config

config = Config("default_env_configs/paper.json")
config.gymify()
config.display()

basic_edps = EDPSEnv()

obs = basic_edps.reset()
initial_debug = basic_edps.debug()

done = False
while not done:
    action = np.unravel_index(obs.argmax(), obs.shape)
    obs, reward, done, debug = basic_edps.step(action)
    pprint(debug)
