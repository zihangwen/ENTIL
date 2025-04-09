import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal


EPS = 1e-6
######################################################################################
class LanguageModule(nn.Module):
    def __init__(self, obj, sound):
        super().__init__()
        self.obj, self.sound = obj, sound
        self.c_entropy = 0
        self.num_samples = 100
        self.sigma = 0.1
        self.P = nn.Parameter(torch.randn(self.obj, self.sound))
        self.Q = nn.Parameter(torch.randn(self.sound, self.obj))
        self.sm = nn.Softmax(dim = -1)
    
    def forward(self):
        smP, smQ = self.sm(self.P), self.sm(self.Q)
        return smP, smQ
    
    def sample(self):
        self.dist_P = normal.Normal(self.P, scale = self.sigma)
        self.dist_Q = normal.Normal(self.Q, scale = self.sigma)
        P_samples = self.dist_P.sample((self.num_samples,))
        Q_samples = self.dist_Q.sample((self.num_samples,))

        smP_samples, smQ_samples = self.sm(P_samples), self.sm(Q_samples)
        payoff, entropy, reward = self.cal_reward(smP_samples, smQ_samples)

        lp_P = self.dist_P.log_prob(P_samples).sum((-1,-2))
        lp_Q = self.dist_Q.log_prob(Q_samples).sum((-1,-2))
        lp = lp_P + lp_Q

        return P_samples, Q_samples, payoff, entropy, reward, lp
    
    def rsample(self):
        epsilon_P = torch.randn((self.num_samples, self.obj, self.sound))
        epsilon_Q = torch.randn((self.num_samples, self.sound, self.obj))
        P_rsamples = self.P + epsilon_P * self.sigma
        Q_rsamples = self.Q + epsilon_Q * self.sigma
        
        smP_rsamples, smQ_rsamples = self.sm(P_rsamples), self.sm(Q_rsamples)
        payoff, entropy, reward = self.cal_reward(smP_rsamples, smQ_rsamples)

        return P_rsamples, Q_rsamples, payoff, entropy, reward
    
    def cal_reward(self, smP, smQ):
        payoff = (smP * smQ.transpose(-1,-2)).sum((-1,-2))
        entropy = - (smP * torch.log(smP + EPS)).sum((-1,-2))
        reward = payoff - self.c_entropy * entropy + 0.01 * torch.randn(payoff.shape)
        return payoff, entropy, reward

############################################################################################
class Solution():
    def __init__(self, obj, sound, lr = 1e-3):
        self.language = LanguageModule(obj, sound)
        self.optimizer = optim.Adam(self.language.parameters(), lr=lr)
        self.game_config = None
        self.training_config = None
        self.hyper_config = None

    def wrapper(self):
        return self.train(), self.idx, self.optimizer.state_dict()
    
    def train(self):
        num_samples = self.game_config["num_samples"]

        technique = self.training_config["technique"]
        c_entropy = self.training_config["c_entropy"]
        period = self.training_config["select_period"]
                
        sigma = self.hyper_config["sigma"]
        
        self.language.num_samples = num_samples
        self.language.c_entropy = c_entropy
        self.language.sigma = sigma
        
        payoff_train = []
        reward_train = []
        for _ in range(period):
            payoff, reward = self._train(technique)
            payoff_train.append(payoff)
            reward_train.append(reward)
        
        return payoff_train, reward_train
    
    def _train(self, technique):
        if technique == "PG":
            return self._train_PG()
        elif technique == "rPG":
            return self._train_rPG()
        elif technique == "gradient":
            return self._train_gradient()
        else:
            raise ValueError("Unknown training technique")

    def _train_PG(self):
        self.optimizer.zero_grad()
        _, _, payoff, _, reward, lp = self.language.sample()
        loss = -((reward - reward.mean()) / (reward.std() + EPS) * lp).mean()
        loss.backward()
        self.optimizer.step()
        return payoff.mean().item(), reward.mean().item()
    
    def _train_rPG(self):
        self.optimizer.zero_grad()
        _, _, payoff, _, reward = self.language.rsample()
        loss = -reward.mean()
        loss.backward()
        self.optimizer.step()
        return payoff.mean().item(), reward.mean().item()

    def _train_gradient(self):
        self.optimizer.zero_grad()
        smP, smQ = self.language()
        payoff, _, reward = self.language.cal_reward(smP, smQ)
        loss = -reward
        loss.backward()
        self.optimizer.step()
        return payoff.item(), reward.mean().item()
