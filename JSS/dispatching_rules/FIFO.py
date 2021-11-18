import random
import numpy as np
# import wandb
import gym
import imageio
import warnings
warnings.filterwarnings("ignore")

from JSS.default_config import default_config

import sys
try: instance_filename = sys.argv[1]
except: instance_filename = 'covid01'

INSTANCE_PATH = f'JSS/instances/{instance_filename}'
OUTPUT_GIF_PATH = f'JSS/gifs/{instance_filename}.gif'

machine2label=['registration', 'x_ray', 'consultation', 'ct_scan', 'dispensary']

def FIFO_worker(default_config):
    # wandb.init(config=default_config)
    # config = wandb.config
    config = default_config
    env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': INSTANCE_PATH})
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
        # print("rendering")
        # temp_image = env.render(machine2label=machine2label).to_image()
        # images.append(imageio.imread(temp_image))
        # print("rendered")
    # print(env.solution.T.argsort().tolist())
    env.reset()
    make_span = env.last_time_step
    # print("Completed simulation")
    # imageio.mimsave(OUTPUT_GIF_PATH, images, format='GIF', fps=2)
    print(make_span)

    # wandb.log({"nb_episodes": 1, "make_span": make_span})


if __name__ == "__main__":
    FIFO_worker(default_config)
