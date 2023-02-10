# 开发者：Bright Fang
# 开发时间：2022/5/8 12:33
import gym
# import mujoco
env = gym.make('Humanoid-v2', render_mode="human")
env = env.unwrapped
for episode in range(200):
    observation = env.reset()   # 环境重置
    print(episode)
    # for timestep in range(100):
    while True:
        # print(timestep)
        env.render()    # 可视化
        action = env.action_space.sample()  # 动作采样
        # ans = env.step(action)
        # print(ans)
        observation_, reward, done, truncated, info = env.step(action)     # 单步交互
        if done:
            # print(observation)
            print('Episode {}'.format(episode))
            break
        observation = observation_
env.close()