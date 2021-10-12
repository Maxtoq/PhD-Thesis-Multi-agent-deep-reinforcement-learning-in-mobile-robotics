import argparse
import torch
import cma
import os
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from gym.spaces import Box
from pathlib import Path
from shutil import copyfile
from tensorboardX import SummaryWriter
from train import make_parallel_env, get_paths, load_scenario_config, make_env
from utils.networks import MLPNetwork


def get_num_params(model):
    return sum(p.numel(model) for p in model.parameters())

def run(config):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)

    # Create model
    # Toy env for getting in and out dimensions
    toy_env = make_parallel_env(config.env_path, 1, config.seed, 
                                config.discrete_action, sce_conf)
    num_in_pol = toy_env.observation_space.shape[0]
    if config.discrete_action:
        num_out_pol = toy_env.action_space.n
    else:
        num_out_pol = toy_env.action_space.shape[0]
    policy = MLPNetwork(num_in_pol, num_out_pol, config.hidden_dim, norm_in=False, 
                        constrain_out=True, discrete_action=config.discrete_action)
    
    # Create the CMA-ES trainer
    es = cma.CMAEvolutionStrategy(get_num_params(policy), 1)

    # Initiate parallel envs
    env = make_parallel_env(config.env_path, es.popsize, config.seed,
                            config.discrete_action, sce_conf)
    
    t = 0
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor

        # Ask for candidate solutions
        solutions = es.ask()

        # Load solutions in model
        

        tell_rewards = np.zeros(len(solutions))
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(sce_conf.nb_agents)]
            print(es.popsize)
            print(torch_obs)
            print(torch_obs.shape)
            return

            # Get actions as torch Variables
            for sol in solutions:
                # Load solution in model
                

                # Get actions
                torch_agent_actions = [policy(obs) for obs in torch_obs]

            # Convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # Rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            if dones[0,0]:
                break

            obs = next_obs
        
        # Get final rewards
        # tell_rewards = 

        # Update CMA-ES model


        # Log rewards
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, 
                            a_ep_rew / config.n_rollout_threads, ep_i)

        # Save model
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            #maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            #maddpg.save(model_cp_path)

    #maddpg.save(model_cp_path)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
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
