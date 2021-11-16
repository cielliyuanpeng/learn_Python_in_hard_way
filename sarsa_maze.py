#导入所需的包
import numpy as np
import matplotlib.pyplot as plt

#设定参数θ的初始值 theta_0，用于确定初始方案
theta_0 = np.array([[np.nan,1,1,np.nan], #S0
                     [np.nan,1,np.nan,1], #S1
                     [np.nan,np.nan,1,1], #S2
                     [1,1,1,np.nan], #S3
                     [np.nan,np.nan,1,1], #S4
                     [1,np.nan,np.nan,np.nan], #S5
                     [1,np.nan,np.nan,np.nan], #S6
                     [1,1,np.nan,np.nan] #S7
                    ])

#将策略参数转化为行动策略的函数
def simple_convert_into_pi_from_theta(theta):
    '''简单的计算百分比'''

    [m,n] = theta.shape #获取θ矩阵大小
    pi = np.zeros((m,n))
    for i in range(0,m):
        pi[i,:] = theta[i,:] / np.nansum(theta[i,:]) #计算百分比
    pi = np.nan_to_num(pi) #将nan转换为0

    return pi

[a,b] = theta_0.shape
Q = np.random.rand(a,b) * theta_0
pi_0 = simple_convert_into_pi_from_theta(theta_0)
#将theta_0乘到各元素上，使墙壁方向值为nan

def get_actions(s,Q,epsilon,pi_0):
    direction = ["up","right","down","left"]

    if np.random.rand()<epsilon:
        next_direction = np.random.choice(direction,p=pi_0[s,:])
         #根据概率pi[s,:]选择direction
    else:
        next_direction = direction[np.nanargmax(Q[s,:])]
       # 根据Q的最大值选direction

    if next_direction == "up":
        next_s = s-3
        action = 0
    elif next_direction == "right":
        next_s = s+1
        action =1
    elif next_direction == "down":
        next_s = s+3
        action =2
    elif next_direction == "left":
        next_s = s-1
        action = 3
    else:
        print("Next direction error")

    return action

def get_s_next(s,a,Q,epsilon,pi_0):
    direction = ["up","right","down","left"]


    next_direction = direction[a]
       # 根据Q的最大值选direction

    if next_direction == "up":
        next_s = s-3
    elif next_direction == "right":
        next_s = s+1
    elif next_direction == "down":
        next_s = s+3
    elif next_direction == "left":
        next_s = s-1
    else:
        print("Next direction error")

    return next_s

#基于Sarsa更新动作价值函数Q
def sarsa(s,a,r,s_next,a_next,Q,eta,gamma):
    if s_next == 8:
        Q[s,a]=Q[s,a]+eta*(r-Q[s,a])
    else:
        Q[s,a]=Q[s,a]+eta*(r+gamma*Q[s_next,a_next]-Q[s,a])
    return Q

# 定义基于Sarsa求解迷宫问题的函数，输出状态，动作的历史记录以及更新后的Q
def goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi):
    s = 0
    a = a_next = get_actions(s,Q,epsilon,pi) #初始动作
    s_a_history = [[0,np.nan]]
    while(1):
        a=a_next
        s_a_history[-1][1]= a
        s_next = get_s_next(s,a,Q,epsilon,pi)
        s_a_history.append([s_next,np.nan])
        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_actions(s_next,Q,epsilon,pi)

        Q = sarsa(s,a,r,s_next,a_next,Q,eta,gamma) #每走一步更新一次Q

        if s_next == 8:
            break
        else:
            s = s_next
    return [s_a_history,Q]

# 通过Sarsa求解迷宫问题
eta = 0.1 #学习率
gamma = 0.9 #时间折扣率
epsilon = 0.5 #随机概率初始值
v = np.nanmax(Q,axis=1)
is_continue = True
episode = 1
while is_continue:
    print("当前回合："+str(episode))

    epsilon = epsilon / 2
    [s_a_history,Q] = goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi_0)
    new_v = np.nanmax(Q,axis=1)
    print(np.sum(np.abs(new_v - v)))
    v = new_v
    print("求解迷宫问题所需步数" + str(len(s_a_history)-1))

    episode = episode + 1
    if episode > 100:
        break
