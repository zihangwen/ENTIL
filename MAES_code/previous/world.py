from previous.core import *
import torch 
import torch.nn as nn

class World(nn.Module):  # multi-agent world
#class World:
    def __init__(self):
        super().__init__()
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 10
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.5
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks
    
    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        """
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        """
        self.integrate_state(p_force)
        """
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        """
        
    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)# * self.dt)
            
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            
            if entity.max_speed is not None:
                speed = torch.linalg.norm(entity.state.p_vel, dim = -1)
                idx = speed > entity.max_speed
                entity.state.p_vel[idx] *= entity.max_speed / speed[idx].unsqueeze(1)
                
            entity.state.p_pos += entity.state.p_vel * self.dt
    
    
    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    torch.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

##########################################################################################
import os
import numpy as np
#import pygame


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class SimpleEnv(object):
    def __init__(self, world, scenario):
        # Rending
        '''
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1

        self.game_font = pygame.freetype.SysFont('', 24)
        '''
        
        self.world = world
        self.scenario = scenario
        self.scenario.reset_world(self.world)
        # Set up the drawing window
        self.renderOn = False
        
    def reset(self):
        self.scenario.reset_world(self.world)
        return self.scenario.observation(self.world.agents[0], self.world)  
    
    def step(self, action):
        self.world.agents[0].action.u = action[0]
        self.world.agents[0].action.c = action[1]
        self.world.step()
        return (self.scenario.observation(self.world.agents[0], self.world),
                self.scenario.reward(self.world.agents[0], self.world))
        
    '''
    def render(self):
        self.screen = pygame.display.set_mode(self.screen.get_size())
        self.renderOn = True

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        self.draw()
        pygame.display.flip()
            
    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))
        screen2 = pygame.Surface([self.width, self.height], pygame.SRCALPHA)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos[0] for entity in self.world.entities]
        cam_range = torch.max(torch.abs(torch.vstack(all_poses))).item()
        
        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            loc = entity.state.p_pos[0].numpy()
            loc = (loc / cam_range) * [self.width, -self.height] // 2 * 0.9
            loc += [self.width // 2, self.height // 2] 
            color = entity.color[0].numpy()#to_color(entity.color.numpy())
            
            for i in range(entity.n):
                x, y = loc[i]
                
                pygame.draw.circle(self.screen, color[i] * 200, (x, y), entity.size * 350)  
                # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet

                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.size * 350, 1)  
                # borders
            
                assert (
                    0 < x < self.width and 0 < y < self.height
                ), f"Coordinates {(x, y)} are out of bounds."

                
                # text
                if isinstance(entity, AgentGroup): 
                    if entity.silent:
                        continue
                    
                    c = entity.state.c[0][i]
                    
                    if torch.all(c == 0):
                        word = "_"
                    #elif self.continuous_actions:
                    #    word = (
                    #        "[" + ",".join([f"{comm:.2f}" for comm in c]) + "]"
                    #    )
                    else:
                        word = alphabet[torch.argmax(c).item()]

                    message = entity.name + str(i) + " sends " + word + "   "
                    message_x_pos = self.width * 0.05
                    message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                    self.game_font.render_to(screen2, 
                                             (message_x_pos, message_y_pos), message, (0, 0, 0))
                    text_line += 1
                 
        self.screen.blit(screen2, (0, 0))


    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
    '''
##################################################################################################
class Scenario(object):
    def make_world(self, n_games = 1, n_agents = 2, n_landmarks = 3, internalize = False):
        world = World()
        # add agents
        world.agents = [AgentGroup() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_group{i}"
            agent.collide = False
            agent.silent = False
            agent.n = n_agents
            ####
            agent.size = 0.03

        # add landmarks
        world.landmarks = [LandmarkGroup() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_group{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.n = n_landmarks
            
        #####
        self.internalize = internalize
        world.n = n_games
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = torch.stack([torch.arange(landmark.n) for _ in range(world.n)])
            landmark.color = to_color(1.0 * landmark.color / landmark.n)
            landmark.state.p_pos = 2 * torch.rand((world.n, landmark.n, world.dim_p)) - 1
            landmark.state.p_vel = torch.zeros((world.n, landmark.n, world.dim_p))
            
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = 2 * torch.rand((world.n, agent.n, world.dim_p)) - 1
            agent.state.p_vel = torch.zeros((world.n, agent.n, world.dim_p))
            agent.action.u = torch.zeros((world.n, agent.n, world.dim_p))
            agent.action.c = torch.zeros((world.n, agent.n, world.dim_c))
            
            # goal agent
            agent.goal_a = torch.stack([torch.arange(agent.n) for i in range(world.n)])
            if not self.internalize:
                agent.goal_a = (agent.goal_a + torch.randint(1, agent.n, (world.n, 1))) % agent.n
            agent.goal_a += agent.n * torch.arange(world.n).unsqueeze(1)
            
            # goal landmark
            agent.goal_b = torch.randint(0, world.landmarks[0].n, (world.n, agent.n))    
            agent.goal_b += world.landmarks[0].n * torch.arange(world.n).unsqueeze(1)
            
            agent.color = to_color(torch.arange(agent.n) / agent.n).repeat(world.n,1,1)
                   
    def observation(self, agents, world):
        # goal observation
        landmarks = world.landmarks[0]
        
        landmark_pos = landmarks.state.p_pos.unsqueeze(1) - agents.state.p_pos.unsqueeze(2)
        landmark_clr = landmarks.color.unsqueeze(1).repeat(1, agents.n, 1, 1)
        goal_a_clr = agents.color.reshape(-1, 3)[agents.goal_a]
        goal_b_clr = landmarks.color.reshape(-1, 3)[agents.goal_b]
        
        goal_a_pos = (agents.state.p_pos.reshape(-1, 2)[agents.goal_a] - agents.state.p_pos)
        goal_b_pos = (landmarks.state.p_pos.reshape(-1, 2)[agents.goal_b] - agents.state.p_pos)
        
        return (torch.cat([agents.state.p_vel, landmark_pos.reshape(world.n, agents.n, -1), 
                           goal_b_clr
                          ], dim = -1),
                agents.action.c[:,[1,0],:])
      
    def reward(self, agents, world):
        goal_a_pos = agents.state.p_pos.reshape(-1, 2)[agents.goal_a].reshape(world.n,-1, 2)
        goal_b_pos = world.landmarks[0].state.p_pos.reshape(-1, 2)[agents.goal_b].reshape(world.n,-1, 2)
        
        return -torch.linalg.norm(goal_b_pos -  goal_a_pos, dim = 2)

def to_color(x):
    x = x.unsqueeze(-1)
    shift = torch.tensor([-1, 0, 1]).reshape(1,1,-1) / 3
    return 0.5 * ((3 * (x - shift) % 3).clip(0,1) - (3 * (x - shift) % 3 - 1).clip(0,1)) + 0.25

