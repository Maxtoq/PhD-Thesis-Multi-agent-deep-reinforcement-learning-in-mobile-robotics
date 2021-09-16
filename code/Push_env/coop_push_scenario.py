import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Action, Entity

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

class Object(Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):
    def __init__(self, nb_objects):
        super(PushWorld, self).__init__()
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.1):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        # Object
        self.objects[obj_i].name = 'object %d' % len(self.objects)
        self.objects[obj_i].color = color
        self.objects[obj_i].size = 0.08
        self.objects[obj_i].initial_mass = 2.0
        # Landmark
        self.landmarks[obj_i].name = 'landmark %d' % len(self.landmarks)
        self.landmarks[obj_i].collide = False
        self.landmarks[obj_i].color = color
        self.landmarks[obj_i].size = 0.01
        # Set initial positions
        self.objects[obj_i].state.p_pos = np.random.uniform(-1, 1, self.dim_p)
        self.landmarks[obj_i].state.p_pos = np.random.uniform(-1, 1, self.dim_p)
        if min_dist is not None:
            while get_dist(
                self.objects[obj_i].state.p_pos, 
                self.landmarks[obj_i].state.p_pos
            ) < min_dist:
                self.landmarks[obj_i].state.p_pos = np.random.uniform(-1, 1, self.dim_p)

    def step(self):
        super(PushWorld, self).step()
        # Randomly add an object
        #if np.random.random() < self.obj_prob:
        #    self.add_object_and_landmark()
        

class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4, collision_pen=10.0):
        world = PushWorld(nb_objects)
        # add agent
        self.nb_agents = nb_agents
        world.agents = [Agent() for i in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = 0.025
            agent.initial_mass = 0.4
            agent.max_speed = 0.1
            agent.color = np.array([0.5,0.0,0.0])
        # Objects and landmarks
        self.nb_objects = nb_objects
        # world attributes
        self.obs_range = obs_range
        self.collision_pen = collision_pen
        # make initial conditions
        self.reset_world(world)
        return world

    def done(self, agent, world):
        # Done if all objects are on their landmarks
        tot_dist = 0.0
        for i, object in enumerate(world.objects):
            tot_dist += get_dist(
                object.state.p_pos, 
                world.landmarks[i].state.p_pos
            )
        if tot_dist == 0.0:
            return True
        return False

    def reset_world(self, world):
        world.reset()
        # set initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Reward = -1 x squared distance between objects and corresponding landmarks
        rew = 0
        for i, object in enumerate(world.objects):
            rew -= get_dist(
                object.state.p_pos, 
                world.landmarks[i].state.p_pos, 
                squared=True
            )

        # Penalty for collision between agents
        for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    rew -= self.collision_pen
        return rew

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
        for entity in world.agents + world.objects:
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


        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_obs)