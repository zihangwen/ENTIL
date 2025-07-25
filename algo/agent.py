import torch
import torch.nn as nn
# from torch.distributions import one_hot_categorical, normal
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Any, Union


@dataclass
class AgentConfig:
    n_input_actor: int
    n_input_critic: int
    n_action: int
    action_type: str = "Box" # "Discrete" or "Box" or "MultiBinary"
    n_hidden: int = 64
        

class AgentModule(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        
        if cfg.action_type == "Discrete":
            self.dist = DistModuleCategorical(cfg)
        elif cfg.action_type == "Box":
            self.dist = DistModuleGaussian(cfg)
        else:
            raise NotImplementedError(f"Action type {cfg.action_type} not implemented")

    # def forward(self, state, action = True, value = True):
    #     a, v = None, None
    #     if action:
    #         a = self.actor(state)
    #     if value:
    #         v = self.critic(state)
    #     return a, v
    
    def sample(self, state, std = None):
        act_features = self.actor(state)
        dist = self.dist(act_features, std)
        action = dist.sample()
        
        entropy = dist.entropy().detach()
        # entropy = -(a_c * torch.log(a_c + 1e-6)).sum(-1)

        log_p = dist.log_prob(action).detach()
        return action, log_p, entropy
    
    # def log_prob(self, action, action_mean, std = 0):
    #     dist = self.dist((1 - std) * action_mean + std / action_mean.shape[-1])
    #     return dist.log_prob(action)

    def evaluate_actions(self, state, action, std = None):
        act_features = self.actor(state)
        dist = self.dist(act_features, std)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action_log_probs.sum(axis=-1), dist_entropy


class DistModuleCategorical(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.t_dist = torch.distributions.OneHotCategorical

        self.out = nn.Sequential(
            nn.Linear(cfg.n_hidden, cfg.n_action),
            nn.Softmax(dim = -1)
        )
        # self.out = nn.Linear(cfg.n_hidden, cfg.n_action)

    def forward(self, x, std):
        x = self.out(x)
        if std is None:
            std = 0
        return self.t_dist((1 - std) * x + std / x.shape[-1])


class DistModuleGaussian(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.t_dist = torch.distributions.Normal
        self.out = nn.Sequential(
            nn.Linear(cfg.n_hidden, cfg.n_action),
            nn.Identity()
        )
        # self.out = nn.Linear(cfg.n_hidden, cfg.n_action)
        # nn.init.orthogonal_(self.out[0].weight, gain=0.01)
        # nn.init.constant_(self.out[0].bias, 0)
        
        self.logstd = nn.Parameter(-0.5 * torch.ones(1, cfg.n_action))

    def forward(self, x, std = None):
        fc_mean = self.out(x)
        if std is None:
            std = self.logstd.exp()
        return self.t_dist(fc_mean, std)


class Actor(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.n_hidden
        self.shared = nn.Sequential(nn.Linear(cfg.n_input_actor, hidden),
                                    nn.Tanh(),
                                    nn.Linear(hidden, hidden), 
                                    nn.Tanh())

    def forward(self, state):
        x = self.shared(state)
        return x
        # raise NotImplementedError("Please finish separating the actor module into discrete and continuous")
        # return [self.action_cont(x), self.action_disc(x)]


class Critic(nn.Module):
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.n_hidden
        self.value = nn.Sequential(nn.Linear(cfg.n_input_critic, hidden),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden), 
                                   nn.Tanh(),
                                   nn.Linear(hidden, 1),
                                   nn.Identity())
        
    def forward(self, state):
        return self.value(state)


# class AgentModuleDiscrete(AgentModuleBase):
#     def __init__(self, cfg: AgentConfig):
#         super().__init__(cfg)
#         self.dist = torch.distributions.OneHotCategorical

#     def sample(self, state, std, entropy = False):
#         a_c = self.actor(state)
#         dist = self.dist((1 - std) * a_c + std / a_c.shape[-1])   
#         a_c = dist.sample()
        
#         if entropy:
#             entropy = dist.entropy().detach()
#         else:
#             entropy = None
#         # entropy = -(a_c * torch.log(a_c + 1e-6)).sum(-1)
#         # entropy = dist.entropy().detach()

#         log_c = dist.log_prob(a_c)
#         return a_c, log_c.detach(), entropy
    
#     def log_prob(self, action, action_mean, std = 0):
#         dist = self.dist((1 - std) * action_mean + std / action_mean.shape[-1])
#         return dist.log_prob(action)


# class AgentModuleContinuous(nn.Module):
#     def __init__(self, cfg: AgentConfig):
#         super().__init__(cfg)
#         self.dist = torch.distributions.Normal

#     def sample(self, state, std):
#         a_u = self.actor(state)
#         dist = self.dist(a_u, std)
#         a_u = dist.sample()
        
#         log_u = dist.log_prob(a_u).sum(-1)
#         return a_u, log_u.detach()
    
#     def log_prob(self, action, action_mean, std):
#         dist = self.dist(action_mean[0], std)
#         return dist.log_prob(action[0]).sum(-1)