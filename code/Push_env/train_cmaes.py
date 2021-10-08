import argparse
import json
import cma
import os
import re
import numpy as np
from tqdm import tqdm
from gym.spaces import Box
from pathlib import Path
from shutil import copyfile
from tensorboardX import SummaryWriter
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv


def run(config):
    pass

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
