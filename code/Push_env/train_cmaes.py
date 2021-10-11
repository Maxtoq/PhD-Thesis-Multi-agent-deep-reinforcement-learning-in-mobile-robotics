import argparse
import cma
import os
import numpy as np
from tqdm import tqdm
from gym.spaces import Box
from pathlib import Path
from shutil import copyfile
from tensorboardX import SummaryWriter
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from train import make_parallel_env, get_paths, load_scenario_config
from utils.networks import MLPNetwork


def run(config):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)

    np.random.seed(config.seed)

    env = make_parallel_env(config.env_path, config.n_rollout_threads, config.seed,
                            config.discrete_action, sce_conf)

    # Create model
    num_in_pol = env.observation_space
    if config.discrete_action:
        num_out_pol = env.action_space.n
    else:
        num_out_pol = env.action_space.shape[0]
    policy = MLPNetwork(num_in_pol, num_out_pol, config.hidden_dim, 
                        constrain_out=True, discrete_action=config.discrete_action)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=1, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--shared_params", action='store_true')
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")

    config = parser.parse_args()

    run(config)
