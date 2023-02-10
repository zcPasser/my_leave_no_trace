import gym
import numpy as np

from baselines import utils
import buffer
from baselines.train import Trainer
# from train import Trainer

MAX_EPISODES = 1500
MAX_STEPS = 800
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300

env_names = ['BipedalWalker-v3']
env = gym.make(env_names[0])

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

buffer = buffer.MemoryBuffer(MAX_BUFFER)
agent = utils.create_agent(S_DIM, A_DIM, A_MAX)
trainer = Trainer(agent, buffer)

total_rewards_per_ep = []
num_steps_per_ep = []

for _ep in range(MAX_EPISODES):
    observation = env.reset()

    r = 0
    i = 0
    for _ in range(MAX_STEPS):
        # env.render()
        state = np.float32(observation[0])
        action = trainer.get_exploration_action(state)

        new_state, reward, done, _, info = env.step(action)
        r += reward
        i += 1

        if done:
            new_state = None
        else:
            new_state = np.float32(new_state)
            buffer.add(state, action, reward, new_state)

        state = new_state
        print(end='\rEpisode : {}, reward : {}'.format(_ep, r))

        trainer.optimize()
        if done:
            break

    # gc.collect()
    total_rewards_per_ep.append(r)
    num_steps_per_ep.append(i + 1)

    if _ep == 0 or (_ep + 1) % 100 == 0:
        trainer.save_models(_ep, './results/Models')

# gc.collect()

np.savetxt("results/1_steps_per_ep.txt", num_steps_per_ep)
np.savetxt("results/1_rewards.txt", total_rewards_per_ep)
