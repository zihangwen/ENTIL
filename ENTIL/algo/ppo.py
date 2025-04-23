import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from envs.util import running
from algo.agent import AgentModule, AgentConfig


class PPO(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
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
        for i in tqdm(range(n), desc = "Training"):
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
            action, _, _ = agent.sample(state, std = 0)
            # a_u, a_c = agent.actor(state)
            # action = [a_u.detach(), nn.functional.one_hot(a_c.argmax(-1), num_classes=a_c.shape[-1])]
            obs, test_reward, terminated, truncated, info = test_env.step(action.numpy())
            state = torch.cat(obs, dim = -1)
            
        return test_reward.mean().item()

    def train(self, n = 1):
        results_train = []
        for i in tqdm(range(n), desc = "Training"):
            results_train += [self._train()]
        return results_train
          
    def _train(self):
        gamma = self.training_config['gamma'] 
        epsilon = self.training_config['epsilon'] 
        c_entropy = self.training_config['c_entropy']
        
        T_max = self.game_config['T_max']
        n_games = self.game_config['N_games']
        
        std_c = self.hyper_config['std_c']

        agent = self.agent
        optimizer = self.optimizer
        env = self.env

        mem = {
            "state" : torch.zeros(n_games, T_max + 1, agent.cfg.n_input_actor),
            "action" : torch.zeros(n_games, T_max, agent.cfg.n_action),
            "reward" : torch.zeros(n_games, T_max, 1),
            "log_prob" : torch.zeros(n_games, T_max, 1)
        }

        norm_advantage = self.norm_advantage
        norm_reward  = self.norm_reward

        mse_loss = self.mse_loss

        ##############################################################################
        obs, info = env.reset()
        state = torch.tensor(obs, dtype=torch.float32)
        mem["state"][:,0] = state
        for t in range(T_max):
            action, log_prob, entropy = agent.sample(state, std_c)
            obs, reward, terminated, truncated, info = env.step(action.numpy())

            state = torch.tensor(obs, dtype=torch.float32)
            mem["state"][:,t + 1] = state

            mem["action"][:,t] = action

            mem["log_prob"][:,t] = log_prob.sum(-1).unsqueeze(-1)
            mem["reward"][:,t] = (torch.tensor(reward, dtype=torch.float32).unsqueeze(-1) - c_entropy * entropy).mean(-1, keepdim = True) 

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
        advantage = (v_target - v_old) # .unsqueeze(2)
        advantage = norm_advantage.add_transform(advantage)

        for idx in (torch.randperm(1 * T_max) % T_max).reshape(-1, T_max):
            # subject to change #
            # (a_u, a_c) = agent.actor(mem["state"][:,idx])
            # value = agent.critic(mem["state"][:,idx])

            # ## loss actor
            # prob_ratio = torch.exp(agent.log_prob((mem["a_u"][:,idx], mem["a_c"][:,idx]),
            #                                       (a_u, a_c), std_u, std_c).unsqueeze(-1) 
            #                        - mem["log_prob"][:,idx])
            
            value = agent.critic(mem["state"][:,idx])
            action_log_probs, _ = agent.evaluate_actions(
                mem["state"][:,idx], mem["action"][:,idx],
                std_c
            )
            prob_ratio = torch.exp(action_log_probs.sum(-1).unsqueeze(-1) - mem["log_prob"][:,idx])
            # ----------------- #

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
