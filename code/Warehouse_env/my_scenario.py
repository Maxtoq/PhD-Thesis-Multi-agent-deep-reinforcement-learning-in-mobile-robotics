import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Action

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

def obj_callback(agent, world):
    action = Action()
    action.u = np.zeros((world.dim_p))
    action.c = np.zeros((world.dim_c))
    return action

class PushScenario(BaseScenario):

    obs_range = 10.0

    def make_world(self, nb_agents=1):
        world = World()
        # add agent
        world.agents = [Agent() for i in range(nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
        # add objects
        nb_objects = 1
        for i in range(nb_objects):
            obj = Agent()
            obj.name = 'object %d' % i
            obj.silent = True
            obj.blind = True
            obj.action_callback = obj_callback
            world.agents.append(obj)
        # add landmarks
        lm = Landmark()
        lm.name = 'landmark'
        lm.collide = False
        world.landmarks.append(lm)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            if 'agent' in agent.name:
                agent.color = np.array([0.5,0.0,0.0])
            elif 'object' in agent.name:
                agent.color = np.array([0.0,0.0,0.5])
        for landmark in world.landmarks:
            landmark.color = np.array([0.75,0.75,0.75])
        # set initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Reward = -1 x squared distance between object and landmark
        rew = get_dist(world.agents[-1].state.p_pos, world.landmarks[0].state.p_pos, squared=True)
        return -rew

    def observation(self, agent, world):
        # Observation:
        #  - Agent state: position, velocity
        #  - Other agents and objects:
        #     - If in sight: [1, relative x, relative y, v_x, v_y]
        #     - If not: [0, 0, 0, 0, 0]
        #  - Landmarks:
        #     - If in sight: [1, relative x, relative y]
        #     - If not: [0, 0, 0]
        # => Full observation dim = 2 + 2 + 5 x (nb_agents_objects - 1) + 3 x (nb_landmarks)
        # All distances are divided by max_distance to be in [0, 1]
        entity_obs = []
        for entity in world.agents:
            if entity is agent: continue
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                entity_obs.append(np.concatenate((
                    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range, entity.state.p_vel
                )))
            else:
                entity_obs.append(np.zeros(5))
        for entity in world.landmarks:
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                entity_obs.append(np.concatenate((
                    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range
                )))
            else:
                entity_obs.append(np.zeros(3))

        # Communication


        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_obs)