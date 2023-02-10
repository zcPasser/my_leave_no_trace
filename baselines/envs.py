import gym

env_names = ['BipedalWalker-v3']

env = gym.make(env_names[0])
obs = env.reset()

MAX_EPISODES = 1500
MAX_STEPS = 800
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300


S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]


env.close()