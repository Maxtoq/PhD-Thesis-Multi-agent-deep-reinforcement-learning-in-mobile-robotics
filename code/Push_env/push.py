import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy

from my_scenario import PushScenario

class RandomPolicy(Policy):
    def __init__(self, env):
        super(RandomPolicy, self).__init__()
        self.env = env

    def action(self, obs):
        obj_pos = obs[5:7]
        if not obj_pos.any():
            return np.random.uniform(-1, 1, self.env.world.dim_p)
        else:
            dir_vec = 0.1 * obj_pos / np.sqrt(np.sum(np.square(obj_pos)))
            return dir_vec

if __name__ == "__main__":
    scenario = PushScenario()
    # Create world
    world = scenario.make_world()

    # Create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.render()

    # Create policies
    policies = [RandomPolicy(env) for i in range(env.n)]

    obs_n = env.reset()
    while True:
        # Get each agent's action
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
            #print(f'Agent {i}: \nobs:{obs_n[i]}\naction:{act_n[i]}')

        # Environment step
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print(reward_n)

        env.render()