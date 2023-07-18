import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

# real-time plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# OS Check
import platform

os = platform.system()

if os == 'Darwin':
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
Lambda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
hidden_size = 128
state_space = 4
action_space = 2

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc_pi = nn.Linear(1024, action_space)
        self.fc_v  = nn.Linear(hidden_size,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        state_lst, action_lst, reward_lst, state_prime_lst, prob_action_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            state, action, reward, state_prime, prob_action, done = transition

            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append([reward])
            state_prime_lst.append(state_prime)
            prob_action_lst.append([prob_action])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        state, action, reward, state_prime, done_mask, prob_action = torch.tensor(state_lst, dtype=torch.float).to(device), \
                                                                        torch.tensor(action_lst).to(device), \
                                                                        torch.tensor(reward_lst).to(device), \
                                                                        torch.tensor(state_prime_lst, dtype=torch.float).to(device), \
                                                                        torch.tensor(done_lst, dtype=torch.float).to(device), \
                                                                        torch.tensor(prob_action_lst).to(device)
        self.data = []
        return state, action, reward, state_prime, done_mask, prob_action
    
    def train_net(self):
        state, action, reward, state_prime, done_mask, prob_action = self.make_batch()

        for i in range(K_epoch):
            td_target = reward + gamma * self.v(state_prime) * done_mask
            delta = td_target - self.v(state)
            # delta의 backword를 막기 위해 detach()를 사용
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * Lambda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()

            pi = self.pi(state, softmax_dim=1)
            # action의 index만 뽑기 위해 gather를 사용
            pi_a = pi.gather(1, action)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_action))  # a/b == exp(log(a)-log(b))


            # PPO의 surrogate loss function
            surr1 = ratio * advantage
            surr2 = torch.clip(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(state), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()




def main():
    env = gym.make('CartPole-v1', render_mode='human')
    model = PPO().to(device)
    score = 0.0
    print_interval = 20

    score_lst = list()
    epi_lst = list()

    for n_epi in range(10000):
        state, info = env.reset()
        done = False
        while not done:
            
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(state).float().to(device))
                m = Categorical(prob)
                action = m.sample().item()
                state_prime, reward, done, truncated, info = env.step(action)                
                model.put_data((state, action, reward/100.0, state_prime, prob[action].item(), done))
                state = state_prime
                score += reward
                if done:
                    break
            model.train_net()
        if n_epi&print_interval==0 and n_epi!=0:
            score_lst.append(score/print_interval)
            epi_lst.append(n_epi)
            print('# of episode :{}, avg score : {:.1f}'.format(n_epi, score/print_interval))
            score = 0.0
    env. close()

# x_val = list()
# y_val = list()

# def animate(i):
#     x_val.append(next(score_lst))
#     y_val.append(next(epi_lst))
#     plt.cla()
#     plt.plot(x_val, y_val)

if __name__ == '__main__':
    print(device)
    main()