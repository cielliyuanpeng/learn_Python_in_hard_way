import pdb
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import gym
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from IPython.display import HTML
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
BATCH_SIZE = 32
CAPACITY = 10000
ENV = 'CartPole-v0'
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 200  # 1次实验中的step数
NUM_EPISODES = 500  # 最大尝试次数
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0,frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        
    anim = animation.FuncAnimation(plt.gcf(),animate,frames=len(frames),
                                   interval=90,repeat=False)

    anim.save(('movie_cartpole_DQN.mp4'))

# 定义用于存储经验的内存类


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        '''将transaction = (state, action, state_next, reward)保存在存储器中'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index+1) % self.capacity  # 将保存的index移动一位

    def sample(self, batch_size):
        '''随机检索Batch_size大小的样本并返回'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 获得CarPole 的2个动作

        #创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        #构建一个神经网络
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)  # 输出网络形状

        #最优化方法的设定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''通过Experience Replay学习网络的连接参数'''

        #1.检查经验池大小
        #1.1经验池太小小于批量数据时不执行操作
        if len(self.memory) < BATCH_SIZE:
            return

        #2.创建小批量数据
        #2.1从经验池获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)
        #pdb.set_trace()
#         2.2将每个变量转换为小批量数据对应的形式
#         得到的transition存储了一个Batch_size的（state, action, state_next, reward）
#         即（state, action, state_next, reward）x batch_size
#         想把它变成小批量数据。
#         设为(state x Batch_size, action x Batch_size, state_next x Batch_size, reward x Batch_size)
        batch = Transition(*zip(*transitions))
#         2.3对每个变量的元素转换为小批量数据对应的形式
#         cat指的是concatenate(连接)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        #3.求Q(s_t,a_t)的值作为监督信号
        #3.1将网络切换到推理模式
        self.model.eval()
        #3.2 求取网络输出的Q(s_t, a_t)
        #self.model(state_batch)输出左右两个Q值
        # 为了求得与此处执行的动作a_t对应的Q值，求取action_batch执行的动作
        # a_t对应的Q值，
        state_action_values = self.model(state_batch).gather(1, action_batch)

        #3.3求取max{Q(s_t + 1, a)}的值
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)
                  ))
        next_state_values = torch.zeros(BATCH_SIZE)

        #求取具有下一状态的index的最大Q值
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()
        # 3.4从Q公式中求取Q(S_t,a_t)作为监督信息
        expect_state_action_values = reward_batch + GAMMA * next_state_values

        # 4.更新连接参数
        self.model.train()

        #4.2计算损失函数
        loss = F.smooth_l1_loss(state_action_values,
                                expect_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        epsilon = 0.5*(1/(episode+1))
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor([[
                random.randrange(self.num_actions)
            ]])

        return action


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]

        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False
        frames = []
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            # print("in step0")
            for step in range(MAX_STEPS):
                #print("in step1")
                if episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())
#                 print(observation_next)
#                 print(done)
                if done:
                    #                     print(done)
                    state_next = None

                    episode_10_list = np.hstack((
                        episode_10_list[1:], step + 1))
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes+1
                else:
                    # print("in step2")
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(
                        state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)
                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next
                #print(done)
                if done:
                    print('%d Episode:Finished after %d steps:10次试验的平均step数=%.1lf' % (
                        episode, step+1, episode_10_list.mean()))
                    break

                if episode_final is True:
                    display_frames_as_gif(frames)
                    break

                if complete_episodes >= 10:
                    print("10轮连续成功")
                    episode_final = True


cartpole_env = Environment()
cartpole_env.run()
