import torch.nn as nn
import torch.optim as optim
from torch.distributions import one_hot_categorical, normal
from previous.world import *
from previous.util import *

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

############################################################################################
class Solution(object):
    def __init__(self):
        self.env = None
        self.agent = AgentModule()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-3)
        
        self.training_config = None
        self.game_config = None
        self.hyper_config = None

        self.norm_advantage = running()
        self.norm_reward = running()
        self.mse_loss = nn.HuberLoss(reduction='none')
        
    def wrapper(self, a, b):
        return self.train_eval(a,b), self.idx, self.optimizer.state_dict(), self.norm_advantage, self.norm_reward
            
    def train_eval(self, n = 1, eval_period = 10):
        results_train, results_eval = [], []
        for i in range(n):
            if i % eval_period == 0:
                results_eval += [self.evaluate()]
            results_train += [self._train()]
        
        return results_train, results_eval
    
    def evaluate(self):
        T_max = self.game_config['T_max']
        
        agent = self.agent
        test_env = self.env
        
        obs = test_env.reset()
        state = torch.cat(obs, dim = -1)
        for t in range(T_max):
            a_u, a_c = agent.actor(state)
            action = [a_u.detach(), nn.functional.one_hot(a_c.argmax(-1), num_classes=a_c.shape[-1])]
            obs, test_reward = test_env.step(action)
            state = torch.cat(obs, dim = -1)
            
        return test_reward.mean().item()

    def train(self, n = 1):
        results_train = []
        for i in range(n):
            results_train += [self._train()]
        return results_train
          
    def _train(self):
        gamma = self.training_config['gamma'] 
        std_u = self.training_config['std_u'] 
        epsilon = self.training_config['epsilon'] 
        c_entropy = self.training_config['c_entropy']
        
        T_max = self.game_config['T_max']
        n_games = self.game_config['N_games']
        n_agents = self.game_config['N_agents']

        
        std_c = self.hyper_config['std_c']

        agent = self.agent
        optimizer = self.optimizer
        env = self.env

        mem = {   "state" : torch.zeros(n_games, T_max + 1, n_agents, 21),
                    "a_u" : torch.zeros(n_games, T_max, n_agents, 2),
                    "a_c" : torch.zeros(n_games, T_max, n_agents, 10),
                 "reward" : torch.zeros(n_games, T_max, 1),
               "log_prob" : torch.zeros(n_games, T_max, n_agents, 1)}

        norm_advantage = self.norm_advantage
        norm_reward  = self.norm_reward

        mse_loss = self.mse_loss

        ##############################################################################
        obs = env.reset()
        state = torch.cat(obs, dim = -1)
        mem["state"][:,0] = state
        for t in range(T_max):
            action, log_prob, entropy = agent.sample(state, std_u, std_c)
            obs, reward = env.step(action)

            state = torch.cat(obs, dim = -1)
            mem["state"][:,t + 1] = state

            mem["a_u"][:,t] = action[0]
            mem["a_c"][:,t] = action[1]

            mem["log_prob"][:,t] = log_prob.unsqueeze(-1)
            mem["reward"][:,t] = (reward - c_entropy * entropy).mean(-1, keepdim = True) 

        ##############################################################################
        v_old = agent.critic((mem["state"])).detach()
        v_target = v_old.clone()

        for t in reversed(range(T_max)):
            v_target[:,t] = mem["reward"][:,t] + gamma * v_target[:,t + 1]

        ##############################################################################
        norm_reward.add(v_target)
        r_togo = norm_reward.rescale(mem["reward"])

        for t in reversed(range(T_max)):
            v_target[:,t] = r_togo[:,t] + gamma * v_target[:,t + 1]

        ##############################################################################
        advantage = (v_target - v_old).unsqueeze(2)
        advantage = norm_advantage.add_transform(advantage)

        for idx in (torch.randperm(1 * T_max) % T_max).reshape(-1, T_max):
            (a_u, a_c) = agent.actor(mem["state"][:,idx])
            value = agent.critic(mem["state"][:,idx])

            ## loss actor
            prob_ratio = torch.exp(agent.log_prob((mem["a_u"][:,idx], mem["a_c"][:,idx]),
                                                  (a_u, a_c), std_u, std_c).unsqueeze(-1) 
                                   - mem["log_prob"][:,idx])

            loss_actor = -torch.minimum(prob_ratio * advantage[:, idx], 
                                        prob_ratio.clip(1 - epsilon, 1 + epsilon) 
                                        * advantage[:, idx]).mean()

            ## loss critic
            v_clip = value.clip(v_old[:, idx] - epsilon, v_old[:, idx] + epsilon)
            loss_critic = torch.maximum(mse_loss(value, v_target[:, idx]),
                                        mse_loss(v_clip, v_target[:, idx])).mean()

            ## total loss
            loss = loss_actor + loss_critic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return reward.mean().item()
