import torch.nn as nn
from torch.distributions import one_hot_categorical, normal
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Any, Union


@dataclass
class AgentConfig:
    n_hidden: int = 64
    n_observation_actor: int = 21
    n_observation_critic: int = 42
    
    n_action_continuous: Union[None, int] = None
    n_action_discrete: Union[None, int] = None


class AgentModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()
        
        self.dist_u = normal.Normal
        self.dist_c = one_hot_categorical.OneHotCategorical
                
    def forward(self, state, action = True, value = True):
        a, v = None, None
        if action:
            a = self.actor(state)
        if value:
            v = self.critic(state)
        return a, v
    
    def sample(self, state, std_u, std_c = 0):
        a_u, a_c = self.actor(state)
        dist_u, dist_c = self.dist_u(a_u, std_u), self.dist_c((1 - std_c) * a_c + std_c / a_c.shape[-1])   
        a_u, a_c =  dist_u.sample(), dist_c.sample()
        
        #entropy = -(a_c * torch.log(a_c + 1e-6)).sum(-1)
        entropy = dist_c.entropy().detach()

        log_u, log_c = dist_u.log_prob(a_u).sum(-1),  dist_c.log_prob(a_c)
        return ([a_u, a_c], (log_u + log_c).detach(), entropy)
    
    def log_prob(self, action, action_mean, std_u, std_c = 0):
        dist_u = self.dist_u(action_mean[0], std_u)
        dist_c = self.dist_c((1 - std_c) * action_mean[1] + std_c / action_mean[1].shape[-1])
        return dist_u.log_prob(action[0]).sum(-1) + dist_c.log_prob(action[1])


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 64
        self.shared = nn.Sequential(nn.Linear(11 + 10, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden), 
                                    nn.ReLU())
        
        self.action = nn.Sequential(nn.Linear(hidden, 2),
                                    nn.Tanh())
        
        self.speak = nn.Sequential(nn.Linear(hidden, 10),
                                   nn.Softmax(dim = -1))
    
    def forward(self, state):
        x = self.shared(state)
        return [self.action(x), self.speak(x)]


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 64
        self.value = nn.Sequential(nn.Linear(2 * (11 + 10), hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden), 
                                   nn.ReLU(),
                                   nn.Linear(hidden, 1))
  
    def forward(self, state):
        x = state.flatten(start_dim = -2)
        return self.value(x)

