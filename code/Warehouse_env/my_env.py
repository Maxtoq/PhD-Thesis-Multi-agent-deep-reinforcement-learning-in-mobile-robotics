import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark

def get_dist(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class Scenario(BaseScenario):

    obs_range = 10.0

    def make_world(self):
        world = World()
        # add agent
        nb_agents = 1
        world.agents = [Agent() for i in range(nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
        # add objects
        nb_objects = 1
        for i in range(nb_objects):
            obj = Landmark()
            obj.name = 'object %d' % i
            obj.movable = True
            world.landmarks.append(obj)
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
            agent.color = np.array([1.0,0.0,0.0])
        for landmark in world.landmarks:
            if 'object' in landmark.name:
                landmark.color = np.array([0.0,0.0,0.75])
            else:
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
        dist = get_dist(world.landmarks[0].state.p_pos, world.landmarks[1].state.p_pos)
        return -dist

    def observation(self, agent, world):
        # Get positions of entities in range
        entity_pos = []
        for entity in world.landmarks:
            if get_dist(agent.p_pos, entity.p_pos) <= self.obs_range:
                entity_pos.append(entity.p_pos)
        return