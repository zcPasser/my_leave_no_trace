import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import envs

env_names = ['BipedalWalker-v3']

params = envs.get_env_agent(env_name=env_names[0])

MAX_EPISODES = params['MAX_EPISODES']
MAX_STEPS = params['MAX_STEPS']
MAX_BUFFER = params['MAX_BUFFER']
MAX_TOTAL_REWARD = params['MAX_TOTAL_REWARD']

env = params['env']

S_DIM = params['S_DIM']
A_DIM = params['A_DIM']
A_MAX = params['A_MAX']

inital_state = params['inital_state']

forward_trainer = params['forward_trainer']
forward_buffer = params['forward_buffer']
reset_trainer = params['reset_trainer']
reset_buffer = params['reset_buffer']


def reset_reward_fn1(state):
    # 取 state（1 * 24）某些维度的值 与 初始状态分布 对应维度 比较
    t1 = torch.Tensor(np.array(state)[[0]])
    # print(t1)
    t2 = torch.Tensor(np.array(inital_state)[[0]])
    # print(t2)
    r0 = float(F.smooth_l1_loss(t1, t2))
    print(r0)
    r1 = float(F.smooth_l1_loss(torch.Tensor(np.array(state)[[9, 11, 13]]), torch.Tensor(np.array(inital_state)[[9, 11, 13]])))
    r2 = float(F.smooth_l1_loss(torch.Tensor(np.array(state)[[4, 6, 8]]), torch.Tensor(np.array(inital_state)[[4, 6, 8]])))
    # r0, r1, r2 分布 代表 state 与 initial_state 在 某几个维度 上的 损失
    r = - (r0 + min(r1, r2))
    # reset_r ：偏向 惩罚，计算 当前state 在 特定维度群上的 损失，用惩罚
    # 结合 CAP？
    return r

def reset_reward_fn2(state):
    # 结合 具体环境，BipedalWalker，船体位置
    r = -5.0 * (env.hull.position[1] < 4.7)
    r += -5.0 * np.abs(state[0])
    # r：负值，偏向惩罚
    return r

# total_rewards_per_ep = []
# num_steps_per_ep = []

def train_with_reset_agent(reset_reward_fn, q_min, save_model_folder, total_rewards_per_ep, num_steps_per_ep):
    for _ep in range(MAX_EPISODES):
        observation = env.reset()[0]

        r = 0
        i = 0
        while i < MAX_STEPS:
            i += 1

            state = np.float32(observation)
            action = forward_trainer.get_exploration_action(state)
            q_val = reset_trainer.critic.forward(Variable(torch.from_numpy(np.float32([state]))),
                                                 Variable(torch.from_numpy(np.float32([action]))))
            # early abort, we switch to reset policy (safe mode)
            if q_val < q_min:
                action = reset_trainer.get_exploitation_action(state)
                i -= 1
            else:
                env.render()

            new_obs, reward, done, _, info = env.step(action)
            r += reward

            if done:
                new_state = None
            else:
                new_state = new_obs
                forward_buffer.add(state, action, reward, new_state)

            observation = new_obs
            print(end='\rEpisode : {}, reward : {}'.format(_ep, r))

            # optimization
            forward_trainer.optimize()
            if done:
                break

        observation = env.reset()[0]
        for i in range(MAX_STEPS):
            state = np.float32(observation)
            action = reset_trainer.get_exploration_action(state)

            new_obs, _, done, _, info = env.step(action)
            reward = reset_reward_fn(new_obs)
            if done:
                new_state = None
            else:
                new_state = np.float32(new_obs)
                reset_buffer.add(state, action, reward, new_state)

            observation = new_obs

            reset_trainer.optimize()
            if done:
                break

        total_rewards_per_ep.append(r)
        num_steps_per_ep.append(i + 1)

        if _ep == 0 or (_ep + 1) % 100 == 0:
            forward_trainer.save_models(_ep, save_model_folder)


