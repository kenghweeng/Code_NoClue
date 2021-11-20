import sys
import time
import ray
import wandb
import random
import numpy as np
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from CustomCallbacks import *
from models import *
from typing import Dict, Tuple
import multiprocessing as mp
import pickle
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf
from env.JssEnv import JssEnv

tf1, tf, tfv = try_import_tf()

# To select for use of specific GPUs:
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,14,15"
_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update `wandb.config`
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(
                k.startswith(item + "/") or k == item
                for item in _config_results):
            config_update[k] = v
        elif any(
                k.startswith(item + "/") or k == item
                for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update

def env_creator(env_config):
    return JssEnv(**env_config)

register_env("jss_env", env_creator)

def train_func(input_file):
    default_config = {
        'env': 'jss_env',
        'seed': 5446,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 3,
        'instance_path': f'instances/{input_file}', 
        # 'instance_path': 'instances/covid01',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': 15, # changed
        'layer_nb': 2,
        'train_batch_size': 15 * 4 * 704, # changed
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,  # TO TUNE
        'sgd_minibatch_size': 33000,
        'layer_size': 319,
        'lr': 0.0006861,  # TO TUNE
        'lr_start': 0.0006861,  # TO TUNE
        'lr_end': 0.00007783,  # TO TUNE
        'clip_param': 0.541,  # TO TUNE
        'vf_clip_param': 26,  # TO TUNE
        'num_sgd_iter': 12,  # TO TUNE
        "vf_loss_coeff": 0.7918,
        "kl_coeff": 0.496,
        'kl_target': 0.05047,  # TO TUNE
        'lambda': 1.0,
        'entropy_coeff': 0.0002458,  # TUNE LATER
        'entropy_start': 0.0002458,
        'entropy_end': 0.002042,
        'entropy_coeff_schedule': None,
        "batch_mode": "truncate_episodes",
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
    }

    wandb.init(config=default_config)
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
        "vf_share_layers": False,
    }
    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }

    config = with_common_config(config)
    config['seed'] = 0
    storage = Storage.options(name="global_storage").remote()
    my_callback = CustomCallbacks()
    config['callbacks'] = lambda : my_callback
    config['train_batch_size'] = config['sgd_minibatch_size']

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]

    config['entropy_coeff'] = config['entropy_start']
    config['entropy_coeff_schedule'] = [[0, config['entropy_start']], [15000000, config['entropy_end']]]

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    stop = {
        "time_total_s": 10 * 60,
    }

    start_time = time.time()
    trainer = PPOTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)
        # wandb.config.update(config_update, allow_val_change=True)
    # trainer.export_policy_model("./models/")
    best_makespan, best_solution = ray.get(storage.get_best_solution.remote())
    
    # try:
    #     fname = config['env_config']['env_config']['instance_path'])
    # except:
    #     print("what")

    return best_makespan, best_solution


if __name__ == "__main__":
    _, input_file = sys.argv
    best_makespan, best_solution = train_func(input_file)
    print(best_solution)
    print(best_makespan)

    with open(f'solutions/{input_file}_sol.pkl', 'wb') as f:
        pickle.dump(best_solution, f)
        
    # Based on stored solution, generate Gantt chart
    





    
