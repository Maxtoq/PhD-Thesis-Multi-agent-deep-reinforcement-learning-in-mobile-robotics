import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy

from my_scenario import PushScenario

class RandomPolicy(Policy):
    def __init__(self, env):
        super(RandomPolicy, self).__init__()
        self.env = env

    def action(self, obs):
        (env.world.dim_p + env.world.dim_c, 1)
        return np.zeros((self.env.world.dim_p + self.env.world.dim_c, 1))

if __name__ == "__main__":
    scenario = PushScenario()
    # Create world
    world = scenario.make_world(2)

    # Create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.render()

    # Create policies
    policies = [RandomPolicy(env) for i in range(env.n)]

    obs_n = env.reset()
    while True:
        continue