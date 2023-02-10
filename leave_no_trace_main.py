from train import train_with_reset_agent, reset_reward_fn1, reset_reward_fn2

import numpy as np

env_names = ['BipedalWalker-v3']

total_rewards_per_ep = []
num_steps_per_ep = []

q_min = -5.0

train_with_reset_agent(reset_reward_fn=reset_reward_fn1,
                       q_min=q_min,
                       save_model_folder='results/Models2',
                       total_rewards_per_ep=total_rewards_per_ep,
                       num_steps_per_ep=num_steps_per_ep)

np.savetxt("results/with_reset_2_rewards.txt", np.array(total_rewards_per_ep))
np.savetxt("results/with_reset_2_steps_per_ep.txt", np.array(num_steps_per_ep))
