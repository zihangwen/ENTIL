import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from pathlib import Path

from envs.util import running
# from algo.agent import AgentModule, AgentConfig
from algo.util import discount_cumsum


class Logger(object):
    def __init__(self, *keys):
        self._logs = {key: [] for key in keys}

    def add(self, key, value):
        if key in self._logs:
            self._logs[key].append(value)
        else:
            raise KeyError(f"Key '{key}' not found in logger.")

    def get_logs(self):
        return self._logs
    
    def save_logs(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._logs, f)


class PPO(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-3)
        # self.a_optimizer = optim.Adam(
        #     list(self.agent.actor.parameters()) + list(self.agent.dist.parameters()),
        #     lr=3e-4
        # )
        # self.c_optimizer = optim.Adam(self.agent.critic.parameters(), lr=1e-3)
        self.logger = Logger(
            "AverageEpRet", "StdEpRet", "EpLen", "NEnv", "AverageVVals", "StdVVals", "KL"
        )
        
        self.training_config = None
        self.game_config = None
        self.hyper_config = None

        # self.norm_advantage = running()
        # self.norm_reward = running()
        # self.mse_loss = nn.HuberLoss(reduction='none')
        
    # def wrapper(self, a, b):
    #     return self.train_eval(a,b), self.idx, self.optimizer.state_dict(), self.norm_advantage, self.norm_reward
    
    # def train_eval(self, n = 1, eval_period = 10):
    #     results_train, results_eval = [], []
    #     for i in tqdm(range(n), desc = "Training"):
    #         if i % eval_period == 0:
    #             results_eval += [self.evaluate()]
    #         results_train += [self._train()]
        
    #     return results_train, results_eval
    
    # def evaluate(self):
    #     T_max = self.game_config['T_max']
        
    #     agent = self.agent
    #     test_env = self.env
        
    #     obs = test_env.reset()
    #     state = torch.cat(obs, dim = -1)
    #     for t in range(T_max):
    #         action, _, _ = agent.sample(state, std = 0)
    #         # a_u, a_c = agent.actor(state)
    #         # action = [a_u.detach(), nn.functional.one_hot(a_c.argmax(-1), num_classes=a_c.shape[-1])]
    #         obs, test_reward, terminated, truncated, info = test_env.step(action.numpy())
    #         state = torch.cat(obs, dim = -1)
            
    #     return test_reward.mean().item()

    def train(self, n : int = 1):
        # torch.manual_seed(10000)
        # np.random.seed(seed)
        for i in tqdm(range(n), desc = "Training"):
            self._train()
        
        return self.logger
          
    def _train(self):
        gamma = self.training_config['gamma'] 
        epsilon = self.training_config['epsilon']
        lam = self.training_config['lam']
        target_kl = self.training_config['target_kl']
        # entropy_coef = self.training_config['entropy_coef']
        train_v_iters = self.training_config['train_v_iters']
        train_a_iters = self.training_config['train_a_iters']

        T_max = self.game_config['T_max']
        n_games = self.game_config['N_games']
        
        # std_c = self.hyper_config['std_c']

        agent = self.agent
        optimizer = self.optimizer
        # a_optimizer = self.a_optimizer
        # c_optimizer = self.c_optimizer
        env = self.env

        mem = {
            "state" : torch.zeros(n_games, T_max + 1, agent.cfg.n_input_actor),
            "action" : torch.zeros(n_games, T_max, agent.cfg.n_action),
            "reward" : torch.zeros(n_games, T_max, 1),
            "log_prob" : torch.zeros(n_games, T_max, 1),
        }

        # norm_advantage = self.norm_advantage
        # norm_reward  = self.norm_reward

        # mse_loss = self.mse_loss

        ##############################################################################
        obs, _ = env.reset()
        ep_ret, ep_len = 0, 0

        state = torch.tensor(obs, dtype=torch.float32)
        mem["state"][:,0] = state
        for t in range(T_max):
            action, log_prob, entropy = agent.sample(state)
            obs, reward, terminated, truncated, info = env.step(action.numpy())
            ep_ret += reward
            ep_len += 1

            state = torch.tensor(obs, dtype=torch.float32)
            mem["state"][:,t + 1] = state

            mem["action"][:,t] = action

            mem["log_prob"][:,t] = log_prob.sum(-1).unsqueeze(-1)
            # mem["reward"][:,t] = (torch.tensor(reward, dtype=torch.float32).unsqueeze(-1) - entropy_coef * entropy).mean(-1, keepdim = True)
            mem["reward"][:,t] = (torch.tensor(reward, dtype=torch.float32).unsqueeze(-1)).mean(-1, keepdim = True)

        # ##############################################################################
        # v_old = agent.critic((mem["state"])).detach()
        # v_target = v_old.clone()

        # for t in reversed(range(T_max)):
        #     v_target[:,t] = mem["reward"][:,t] + gamma * v_target[:,t + 1]

        # ##############################################################################
        # norm_reward.add(v_target)
        # r_togo = norm_reward.rescale(mem["reward"])

        # for t in reversed(range(T_max)):
        #     v_target[:,t] = r_togo[:,t] + gamma * v_target[:,t + 1]

        # ##############################################################################
        # advantage = (v_target - v_old) # .unsqueeze(2)
        # advantage = norm_advantage.add_transform(advantage)

        vals = agent.critic((mem["state"])).detach()
        rews = torch.cat([mem["reward"], vals[:,-1].unsqueeze(-1)], dim=1)
        deltas = rews[:,:-1] + gamma * vals[:,1:] - vals[:,:-1]
        adv_buf = discount_cumsum(deltas, gamma * lam)
        advantage = (adv_buf - adv_buf.mean()) / (adv_buf.std())
        ret_buf = discount_cumsum(rews, gamma)[:,:-1]

        # ----------------- #
        for _ in range(train_a_iters):
            action_log_probs, _ = agent.evaluate_actions(
                mem["state"][:,:-1], mem["action"]
            )
            delta_log_probs = action_log_probs.unsqueeze(-1) - mem["log_prob"]
            prob_ratio = torch.exp(delta_log_probs)
            approx_kl = - delta_log_probs.mean().item()

            if approx_kl > 1.5 * target_kl:
                break

            loss_actor = -torch.minimum(prob_ratio * advantage, 
                                        prob_ratio.clip(1 - epsilon, 1 + epsilon) 
                                        * advantage).mean()
            optimizer.zero_grad()
            loss_actor.backward()
            optimizer.step()
        
        # ----------------- #
        for _ in range(train_v_iters):
            ## loss critic
            value = agent.critic(mem["state"][:,:-1])
            # v_clip = value.clip(v_old - epsilon, v_old + epsilon)
            # loss_critic = torch.maximum(mse_loss(value, ret_buf),
            #                             mse_loss(v_clip, ret_buf)).mean()
            loss_critic = ((value - ret_buf) ** 2).mean()

            optimizer.zero_grad()
            loss_critic.backward()
            optimizer.step()

        self.logger.add("AverageEpRet", ep_ret.mean().item())
        self.logger.add("StdEpRet", ep_ret.std().item())
        self.logger.add("EpLen", ep_len)
        self.logger.add("NEnv", n_games)
        self.logger.add("AverageVVals", vals.mean().item())
        self.logger.add("StdVVals", vals.std().item())
        self.logger.add("KL", approx_kl)
