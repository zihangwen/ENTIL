import torch.nn as nn
from torch.distributions import one_hot_categorical, normal
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Any, Union


@dataclass
class AgentConfig:
    n_hidden: int = 64
    n_observation_actor: int = 21
    n_observation_critic: int = 42
    
    modality: str = "discrete"
    n_action: Union[None, int] = None


class AgentModuleBase(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        
        self.dist_u = normal.Normal
        self.dist_c = one_hot_categorical.OneHotCategorical
    
    def forward(self, state, action = True, value = True):
        a, v = None, None
        if action:
            a = self.actor(state)
        if value:
            v = self.critic(state)
        return a, v
    
    def sample(self, state, std_c = 0):
        raise NotImplementedError

    def log_prob(self, action, action_mean, std = 0):
        raise NotImplementedError


class AgentModuleDiscrete(AgentModuleBase):
    def __init__(self, cfg: AgentConfig):
        super().__init__(cfg)
                    
    def sample(self, state, std, entropy = False):
        a_c = self.actor(state)
        dist_c = self.dist_c((1 - std) * a_c + std / a_c.shape[-1])   
        a_c = dist_c.sample()
        
        if entropy:
            entropy = dist_c.entropy().detach()
        else:
            entropy = None
        # entropy = -(a_c * torch.log(a_c + 1e-6)).sum(-1)
        # entropy = dist_c.entropy().detach()

        log_c = dist_c.log_prob(a_c)
        return a_c, log_c.detach(), entropy
    
    def log_prob(self, action, action_mean, std = 0):
        dist_c = self.dist_c((1 - std) * action_mean + std / action_mean.shape[-1])
        return dist_c.log_prob(action)


class AgentModuleContinuous(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__(cfg)
                
    def sample(self, state, std):
        a_u = self.actor(state)
        dist_u = self.dist_u(a_u, std)
        a_u = dist_u.sample()
        
        log_u = dist_u.log_prob(a_u).sum(-1)
        return a_u, log_u.detach()
    
    def log_prob(self, action, action_mean, std):
        dist_u = self.dist_u(action_mean[0], std)
        return dist_u.log_prob(action[0]).sum(-1)


class Actor(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.n_hidden
        self.shared = nn.Sequential(nn.Linear(cfg.n_observation_actor, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden), 
                                    nn.ReLU())
        
        if cfg.n_action_continuous is not None:
            self.action_cont = nn.Sequential(
                nn.Linear(hidden, cfg.n_action_continuous),
                nn.Tanh()
            )
        elif cfg.n_action_discrete is not None:
            self.action_disc = nn.Sequential(
                nn.Linear(hidden, cfg.n_action_discrete),
                nn.Softmax(dim = -1)
            )
    
    def forward(self, state):
        x = self.shared(state)
        raise NotImplementedError("Please finish separating the actor module into discrete and continuous")
        # return [self.action_cont(x), self.action_disc(x)]


class Critic(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.n_hidden
        self.value = nn.Sequential(nn.Linear(cfg.n_observation_critic, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden), 
                                   nn.ReLU(),
                                   nn.Linear(hidden, 1))
  
    def forward(self, state):
        x = state.flatten(start_dim = -2)
        return self.value(x)

