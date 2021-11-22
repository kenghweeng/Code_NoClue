import random
import numpy as np
import wandb
import gym
import imageio
import warnings
warnings.filterwarnings("ignore")

from JSS.default_config import default_config
from JSS.env import JssEnv

import sys
try: instance_filename = sys.argv[1]
except: instance_filename = 'covid01'

INSTANCE_PATH = f'JSS/instances/{instance_filename}'
OUTPUT_GIF_PATH = f'JSS/images/{instance_filename}_FIFO.gif'
machine2label=['daily_rounds', 'x_ray', 'consultation', 'ct_scan', 'dispensary']

def FIFO_worker(default_config):
    # wandb.init(config=default_config) # comment out here for non-use of wandb
    # config = wandb.config # comment out here for non-use of wandb
    config = default_config
    # env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': config['instance_path']})
    env = JssEnv(env_config={'instance_path': INSTANCE_PATH})
    env.seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    done = False
    state = env.reset()
    images = []
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))
        remaining_time = reshaped[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time)
        assert legal_actions[FIFO_action]
        state, reward, done, _ = env.step(FIFO_action)

        print("rendering")
        temp_image = env.render(machine2label=machine2label).to_image()
        images.append(imageio.imread(temp_image))
        print("rendered")


    env.reset()
    make_span = env.last_time_step
    # wandb.log({"nb_episodes": 1, "make_span": make_span}) # comment out here for non-use of wandb

    print("Completed simulation")
    imageio.mimsave(OUTPUT_GIF_PATH, images, format='GIF', fps=2) # uncomment for generation of GIF

if __name__ == "__main__":
    FIFO_worker(default_config)
