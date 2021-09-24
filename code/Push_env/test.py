import argparse
import torch
import sys
import os
import numpy as np

from torch.autograd import Variable
from algorithms.maddpg import MADDPG
from train import make_env

def run(config):
    # Load model
    if not os.path.exists(config.model_cp_path):
        sys.exit("Path to the model checkpoint %s does not exist" % 
                    config.model_cp_path)
    maddpg = MADDPG.init_from_save(config.model_cp_path)
    maddpg.prep_rollouts(device='cpu')

    # Create environment
    env = make_env(config.env_path, discrete_action=config.discrete_action, 
                           sce_conf_path=config.sce_conf_path)
    env.seed(config.seed * 1000)
    #np.random.seed(config.seed * 1000)

    for ep_i in range(config.n_episodes):
        obs = env.reset()
        rew = 0
        for step_i in range(config.episode_length):
            # Get each agent's action
            act_n = []
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[a]).unsqueeze(0),
                                requires_grad=False)
                        for a in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().squeeze() for ac in torch_agent_actions]
            #print(agent_actions)
            #actions = [agent_actions[0][i] for i in range(maddpg.nagents)]
            
            # Environment step
            next_obs, rewards, dones, infos = env.step(actions)
            rew += rewards[0]

            env.render()

            if dones[0]:
                break
            obs = next_obs
        
        print(f'Episode {ep_i + 1} finished after {step_i + 1} steps with return {rew}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_cp_path", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument("--seed",default=1, type=int, help="Random seed")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=250, type=int)
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()

    run(config)