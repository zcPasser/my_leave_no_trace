import gym

import buffer
import utils
from baselines.train import Trainer


def get_env_agent(env_name) -> dict:
    info = dict()

    env = gym.make(env_name)
    obs = env.reset()

    info['env'] = env

    # S_DIM = env.observation_space.shape[0]
    info['S_DIM'] = env.observation_space.shape[0]
    # A_DIM = env.action_space.shape[0]
    info['A_DIM'] = env.action_space.shape[0]
    # A_MAX = env.action_space.high[0]
    info['A_MAX'] = env.action_space.high[0]
    # MAX_EPISODES = 1500
    info['MAX_EPISODES'] = 500
    # MAX_STEPS = 800
    info['MAX_STEPS'] = 800
    # MAX_BUFFER = 1000000
    info['MAX_BUFFER'] = 1000000
    # MAX_TOTAL_REWARD = 300
    info['MAX_TOTAL_REWARD'] = 300

    # inital_state = obs[0]
    info['inital_state'] = obs[0]
    # 经验池
    info['forward_buffer'] = buffer.MemoryBuffer(info['MAX_BUFFER'])
    info['reset_buffer'] = buffer.MemoryBuffer(info['MAX_BUFFER'])
    # 前向智能体 & 重置智能体
    info['forward_agent'] = utils.create_agent(info['S_DIM'], info['A_DIM'], info['A_MAX'])
    info['reset_agent'] = utils.create_agent(info['S_DIM'], info['A_DIM'], info['A_MAX'])
    # trainer
    info['forward_trainer'] = Trainer(info['forward_agent'], info['forward_buffer'])
    info['reset_trainer'] = Trainer(info['reset_agent'], info['reset_buffer'])

    return info


