import argparse
import torch
import json
import imp
import sys
import os
import re
import numpy as np
from tqdm import tqdm
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from shutil import copyfile
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()

def make_env(scenario_path, sce_conf_path=None, benchmark=False, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_path   :   path of the scenario script
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = imp.load_source('', scenario_path).Scenario()
    # create world
    conf = {}
    if sce_conf_path is not None:
        with open(sce_conf_path) as cf:
            conf = json.load(cf)
            print('Special config for scenario:', scenario_path)
            print(conf, '\n')
    world = scenario.make_world(**conf)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data, 
                            done_callback=scenario.done if hasattr(scenario, "done") 
                            else None, discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, 
                            done_callback=scenario.done if hasattr(scenario, "done")
                            else None, discrete_action=discrete_action)
    return env

def make_parallel_env(env_path, n_rollout_threads, seed, discrete_action, 
                      sce_conf_path=None):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_path, discrete_action=discrete_action, 
                           sce_conf_path=sce_conf_path)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def save_scenario_cfg(config, save_dir):
    if config.sce_conf_path is not None:
        copyfile(config.sce_conf_path, save_dir / 'sce_config.json')

def run(config):
    start_ep = 0
    # Get environment name from script path
    env_name = re.findall("\/?([^\/.]*)\.py", config.env_path)[0]
    # Get path of the run directory
    model_dir = Path('./models') / env_name / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    elif config.run_name is not None:
        run_dir = model_dir / config.run_name
        if not run_dir.exists():
            sys.exit('Error: run directory %s does not exist.' % run_dir)
        model_cp_path = run_dir / 'model.pt'
        if not model_cp_path.exists():
            sys.exit('Error: model checkpoint path %s does not exist. \
                Unable to load the run.' % model_cp_path)
        curr_run = config.run_name
        with open(str(run_dir / 'logs' / 'ep_nb.txt')) as f:
            start_ep = int(f.read()) + 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    model_cp_path = run_dir / 'model.pt'
    log_dir = run_dir / 'logs'
    if not log_dir.exists():
        os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_path, config.n_rollout_threads, config.seed,
                            config.discrete_action, config.sce_conf_path)
    
    save_scenario_cfg(config, run_dir)

    if config.run_name is None:
        maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                    adversary_alg=config.adversary_alg,
                                    tau=config.tau,
                                    lr=config.lr,
                                    hidden_dim=config.hidden_dim,
                                    shared_params=config.shared_params)
    else:
        maddpg = MADDPG.init_from_save(model_cp_path)

    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in tqdm(range(start_ep, config.n_episodes, config.n_rollout_threads)):
        #print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                ep_i + 1 + config.n_rollout_threads,
        #                                config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            if dones[0,0]:
                break
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
            ) 
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, 
                            a_ep_rew / config.n_rollout_threads, ep_i)
        # Save ep number
        with open(str(log_dir / 'ep_nb.txt'), 'w') as f:
            f.write(str(ep_i))

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(model_cp_path)

    maddpg.save(model_cp_path)
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
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=250, type=int)
    parser.add_argument("--steps_per_update", default=1000, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--shared_params", action='store_true')
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")

    config = parser.parse_args()

    run(config)