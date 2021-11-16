import numpy as np
import matplotlib.pyplot as pyplot
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
#常量设定
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 1000

NUM_PROCESSES = 16 #16个agent
NUM_ADVANCED_STEP = 5 #设置提前计算奖励总和的步数
# 用于计算A2C的误差函数的常量设计
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

#存储类定义
class RolloutStorage(object):
    def __init__(self,num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps+1, num_processes, 4)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        #存储折扣奖励总和
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        self.index = 0#要insert的索引

    def insert(self, current_obs, action, reward, mask):
        '''存储transition到下一个index'''
        self.observations[self.index+1].copy_(current_obs)
        self.masks[self.index+1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1)%NUM_ADVANCED_STEP #更新索引

    def after_update(self):
        '''当Advantage的step数已经完成时，最新的一个存在index0'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''计算Advantage步骤中每个步骤的折扣奖励总和'''

        #从第五步反向计算

        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step+1] * \
                GAMMA * self.masks[ad_step+1] + self.rewards[ad_step] 
        
#A2C深度神经网络的构建
class Net(nn.Module):
    def __init__(self,n_in,n_mid,n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in,n_mid)
        self.fc2 = nn.Linear(n_mid,n_mid)
        self.actor = nn.Linear(n_mid, n_out)
        self.critic = nn.Linear(n_mid,1)

    def forward(self,x):
        '''定义网络前向运算'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)

        return critic_output,actor_output

    def act(self,x):
        '''按概率求状态x的动作'''
        value, actor_output = self(x)
        #在动作类型方向计算softmax,dim=1
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        #dim = 1的动作类型方向的概率计算
        return action

    def get_value(self,x):
        '''从状态x获得状态价值'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self,x,actions):
        '''从状态x获取状态值，记录实际动作的对数概率和熵'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)
        #使用dim=1在动作类型方向上计算

        action_log_probs = log_probs.gather(1, actions)
        #求实际动作的log_probs
        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs*probs).sum(-1).mean()

        return value, action_log_probs, entropy


class Brain(object):
    def __init__(self,actor_critic):
        self.actor_critic = actor_critic #actor_critic是一个Net神经网络
        self.optimizer = optim.Adam(self.actor_critic.parameters(),lr=0.01)

    def update(self, rollouts):
        '''对使用Advantage计算的所有5个步骤进行更新'''
        obs_shape = rollouts.observations.size()[2:] #torch.Size([4,48,84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1,4),
            rollouts.actions.view(-1,1)
        )

        values = values.view(num_steps, num_processes,1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        #Advantage的计算（动作价值-状态价值）
        advantages = rollouts.returns[:-1] - values

        #计算critic的损失loss
        value_loss = advantages.pow(2).mean()

        #计算Actor的gain，然后添加负号以其作为loss
        action_gain = (action_log_probs*advantages.detach()).mean()

        #误差函数计算总和
        total_loss = (value_loss*value_loss_coef-action_gain-entropy*entropy_coef)

        #更新连接参数
        self.actor_critic.train() 
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),max_grad_norm)
        self.optimizer.step()

class Environment:
    def run(self):

        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]
        #生成所有Agent共享的Brain
        n_in = envs[0].observation_space.shape[0]
        n_out = envs[0].action_space.n
        n_mid = 32
        actor_critic = Net(n_in,n_mid,n_out)

        global_brain=Brain(actor_critic)

        #生成存储变量
        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES,obs_shape)
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)
        episode_rewards = torch.zeros([NUM_PROCESSES,1])
        final_rewards = torch.zeros([NUM_PROCESSES,1])
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])
        reward_np = np.zeros([NUM_PROCESSES,1])
        done_np = np.zeros([NUM_PROCESSES, 1])
        each_step = np.zeros(NUM_PROCESSES)
        episode = 0

        #初始状态
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()
        current_obs = obs

        rollouts.observations[0].copy_(current_obs)

        #将当前状态保存到对象rollouts的第一个状态进行advanced学习
        rollouts.observations[0].copy_(current_obs)

        #运行循环
        for j in range(NUM_EPISODES*NUM_PROCESSES):
            #计算advanced学习的每个step数
            for step in range(NUM_ADVANCED_STEP):
                #求取动作
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                    actions = action.squeeze(1).numpy()

                    #运行一步
                    for i in range(NUM_PROCESSES):
                        obs_np[i],reward_np[i],done_np[i],_ = envs[i].step(actions[i])

                        #判断当前episode是否终止，是否有下一个状态
                        if done_np[i]:
                            if i==0:
                                print('%d Episode: Finished after %d steps'%(episode,each_step[i]+1))

                                episode+=1

                            if each_step[i]<195:
                                reward_np[i] = -1.0
                            else:
                                reward_np[i] = 1.0
                            
                            each_step[i] = 0
                            obs_np[i] = envs[i].reset()
                        else:
                            reward_np[i]=0.0
                            each_step[i]+=1
                    reward = torch.from_numpy(reward_np).float()
                    episode_rewards += reward
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done_np]
                    )
                    final_rewards += (1-masks) * episode_rewards
                    episode_rewards *= masks
                    current_obs *= masks
                    obs = torch.from_numpy(obs_np).float()
                    current_obs = obs
                    rollouts.insert(current_obs,action.data,reward,masks)
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()

            rollouts.compute_returns(next_value)    

            global_brain.update(rollouts)
            rollouts.after_update()

            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print("连续成功")
                break






cartpole_env = Environment()
cartpole_env.run()

