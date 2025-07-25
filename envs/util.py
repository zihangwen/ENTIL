import torch 
import torch.nn as nn
from torch.distributions import normal
import math

class running(object):
    def __init__(self, gamma_m = 0.99, gamma_v = 0.99):
        self.gamma_m = gamma_m
        self.gamma_v = gamma_v
        
        self.m = 0
        self.v = 0
        self.t = 0
        
    def add(self, data):
        self.m = self.gamma_m * self.m + (1 - self.gamma_m) * data.mean()
        self.v = self.gamma_v * self.v + (1 - self.gamma_v) * data.square().mean()
        self.t += 1

    def rescale(self, data):
        m = self.m / (1 - self.gamma_m**self.t)
        v = self.v / (1 - self.gamma_v**self.t)
        #return data / (v - m.square()).sqrt().clip(0.01, 1e3)
        return data / v.sqrt().clip(0.01, 1e3)

    def scale(self, data):
        m = self.m / (1 - self.gamma_m**self.t)
        v = self.v / (1 - self.gamma_v**self.t)
        #return data * (v - m.square()).sqrt().clip(0.01, 1e3)
        return data * v.sqrt().clip(0.01, 1e3)
    
    def transform(self, data):
        m = self.m / (1 - self.gamma_m**self.t)
        v = self.v / (1 - self.gamma_v**self.t)
        return (data - m) / (v - m.square()).sqrt().clip(0.01, 1e3)

    def inverse_transform(self, data):
        m = self.m / (1 - self.gamma_m**self.t)
        v = self.v / (1 - self.gamma_v**self.t)
        return data * (v - m.square()).sqrt().clip(0.01, 1e3) + m
    
    def add_transform(self, data):
        self.add(data)
        return self.transform(data)

##################################################################################################
class Random_Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.mean = nn.Linear(n_in, n_out)
        self.tune_std = True #tune_std

        if self.tune_std:
            self.std = nn.Linear(n_in, n_out)
            for param in self.std.parameters():
                nn.init.constant_(param, 0.05)
        
            self.w_dist = normal.Normal(self.mean.weight, self.std.weight)
            self.b_dist = normal.Normal(self.mean.bias.unsqueeze(0), self.std.bias)
        else:
            self.std = 0.05
        
            self.w_dist = normal.Normal(self.mean.weight, self.std)
            self.b_dist = normal.Normal(self.mean.bias.unsqueeze(0), self.std)

        self.w, self.b = None, None
        
    def forward(self, x): 
        return self.mean(x)
    
    def log_prob(self):
        return (self.w_dist.log_prob(self.w).sum(dim = (1,2)) 
                + self.b_dist.log_prob(self.b).sum(dim = (1,2))).unsqueeze(-1)
            
    def sample(self, x):
        assert len(x.shape) == 3
        
        self.sample_p(x.shape[0])
        return self.sample_x(x)
        
    def sample_x(self, x):
        return torch.bmm(x, self.w.transpose(1,2)) + self.b
        
    def sample_p(self, n_samples):
        if self.tune_std:
            for param in self.std.parameters():
                param.data = param.data.clamp(min=1e-6)
            
        self.w = self.w_dist.sample((n_samples, ))
        self.b = self.b_dist.sample((n_samples, ))
        
    def check_grad(self):
        with torch.no_grad():
            w_grad = ((self.w - self.mean.weight) / self.std ** 2).mean(0)
            b_grad = ((self.b - self.mean.bias) / self.std ** 2).mean(0)   
        return w_grad, b_grad

##################################################################################################
class PQ_Linear(nn.Module):
    def __init__(self, n_width, n_in, n_out):
        super().__init__()
        self.p_mean = nn.Linear(n_in, n_out)
        self.q_mean = nn.Linear(n_in, n_out)
        self.q_to_p()
        
        self.p_std = 0.05
        self.q_std = 0.05
        
        self.set_width(n_width)
        self.mutate()
        
    def set_width(self, n_width):
        self.w = torch.empty(n_width, self.q_mean.out_features, self.q_mean.in_features)
        self.b = torch.empty(n_width, 1, self.q_mean.out_features)
        self.mutate()
        
    def forward(self, x): 
        return self.q_mean(x)
    
    def forward_p(self, x): 
        return self.p_mean(x)
    
    def sample(self, x):
        return torch.bmm(x, self.w.transpose(1,2)) + self.b
    
    def log_ratio(self):
        # unnormalized log prob
        p_var = self.p_std ** 2
        q_var = self.q_std ** 2
        
        with torch.no_grad():
            log_p = ((self.w - self.p_mean.weight).square().sum(dim = (1,2)) / p_var) 
            log_p += ((self.b - self.p_mean.bias).square().sum(dim = (1,2)) / p_var) 
            log_p = -math.log(self.p_std) - log_p / 2
            
            log_q = ((self.w - self.q_mean.weight).square().sum(dim = (1,2)) / q_var) 
            log_q += ((self.b - self.q_mean.bias).square().sum(dim = (1,2)) / q_var) 
            log_q = -math.log(self.q_std) - log_q / 2
            
        return log_p - log_q
    
    def select(self, fitness, prob_ratio):
        with torch.no_grad():
            # d/da p/q = dp/q = p/q dp/p
            assert (fitness.shape[0] == self.w.shape[0] and 
                    prob_ratio.shape[0] == self.w.shape[0])
            
            prob_ratio = prob_ratio / prob_ratio.sum()#.clip(0.5, 2.0)
            prob_ratio = prob_ratio.reshape(-1,1,1)
            fitness = fitness.reshape(-1,1,1)
            
            # weight gradient
            dp_over_p = (self.w - self.p_mean.weight) / self.p_std ** 2
            self.p_mean.weight.grad = (fitness * prob_ratio * dp_over_p).mean(0)
            
            # bias gradient
            dp_over_p = (self.b - self.p_mean.bias) / self.p_std ** 2
            self.p_mean.bias.grad = (fitness * prob_ratio * dp_over_p).mean(0).squeeze(0)
        
    def mutate(self):
        with torch.no_grad():
            l,w,h = self.w.shape
            self.w = self.q_std * torch.randn(self.w.shape)
            self.w += self.q_mean.weight

            self.b = self.q_std * torch.randn(self.b.shape) 
            self.b += self.q_mean.bias
             
    def q_to_p(self):
        for p, q in zip(self.p_mean.parameters(), self.q_mean.parameters()):
            q.data = p.data.clone()
    
    def prob_ratio(self):
        p_var = self.p_std ** 2
        q_var = self.q_std ** 2
        
        with torch.no_grad():
            prob_ratio = ((self.w - self.p_mean.weight).square().sum(dim = (1,2)) / p_var) 
            prob_ratio += ((self.b - self.p_mean.bias).square().sum(dim = (1,2)) / p_var) 

            prob_ratio -= ((self.w - self.q_mean.weight).square().sum(dim = (1,2)) / q_var)
            prob_ratio -= ((self.b - self.q_mean.bias).square().sum(dim = (1,2)) / q_var)

            prob_ratio = self.p_std / self.q_std * torch.exp(-prob_ratio / 2)
        return prob_ratio
    
    def dist(self):
        dist = 0
        for p, q in zip(self.p_mean.parameters(), self.q_mean.parameters()):
            dist += ((p.data - q.data)**2).sum().item()
        return dist / self.p_std ** 2

##################################################################################################
class ReplayBuffer():
    def __init__(self, n_size, n_agents, state_dim, action_dim):
        self.state = torch.zeros(n_size, n_agents, state_dim)
        self.next_state = torch.zeros(self.state.shape)
        self.action = torch.zeros(n_size, n_agents, action_dim)
        self.reward = torch.zeros(n_size, 1)
         
        self.ptr = 0
        self.count = 0
        self.size = n_size
        
    def add(self, state, next_state, action, reward):
        n_len = state.shape[0]
        end = min(self.ptr + n_len, self.size)
        
        self.state[self.ptr:end] = state[:end - self.ptr]
        self.next_state[self.ptr:end] = next_state[:end - self.ptr]
        self.action[self.ptr:end] = action[:end - self.ptr]
        self.reward[self.ptr:end] = reward[:end - self.ptr]
        
        self.ptr = (self.ptr + n_len) % self.size 
        
        if end == self.size:
            self.count += 1
            self.add(state[-self.ptr:], next_state[-self.ptr:], 
                     action[-self.ptr:], reward[-self.ptr:])
    
    def sample(self, n):
        idx = torch.randint(self.size, size = (n,) )
        #print(max(idx))
        return (self.state[idx],
                self.next_state[idx],
                self.action[idx],
                self.reward[idx])
