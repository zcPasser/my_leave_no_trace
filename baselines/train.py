import torch
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable

import utils

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

class Trainer:
    def __init__(self, agent, buffer):
        self.buffer = buffer
        self.iter = 0
        self.agent = agent
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.agent['a_dim'])

        self.actor = self.agent['actor']
        self.target_actor = self.agent['target_actor']
        # 对 actor网络 参数 进行 优化
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = self.agent['critic']
        self.target_critic = self.agent['target_critic']
        # 对 actor网络 参数 进行 优化
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

    def get_exploitation_action(self, s):
        """
        gets the action from target actor added with exploration noise
        :param s: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # pytorch两个基本对象：Tensor（张量）和Variable（变量）
        # 其中，tensor不能反向传播，variable可以反向传播。
        # Variable 是计算图的一部分, 可以用来传递误差就好.
        s = Variable(torch.from_numpy(s))
        # detach: 从计算图中脱离出来。
        # 返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False。
        a = self.target_actor.forward(s).detach()

        return a.data.numpy()

    def get_exploration_action(self, s):
        """
        gets the action from actor added with exploration noise
        :param s: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(s))
        action = self.actor.forward(state).detach()
        # exploration 动作：actor网络 + 噪声OU过程(结合a_lim)
        new_action = action.data.numpy() + (self.noise.sample() * self.agent["a_lim"])
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.buffer.sample(BATCH_SIZE)

        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        # squeeze()函数的功能是维度压缩。
        # 返回一个tensor（张量），其中 input 中大小为1的所有维都已删除。
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = torch.squeeze(r1 + GAMMA * next_val)
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        # 构建假设函数
        # 构建损失函数
        # 构建优化器(假设函数的权值)
        # 梯度清零 weights.grad=None
        # 求出更新 weights.grad 的值
        # 根据新的 weights.grad 值更新迭代 weights 值
        # 计算假设函数的值
        # 计算损失
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        # 使用torch.tensor()对 自变量x xx 进行构建的时候，曾经对参数requires_grad=True进行设置为True，
        # 一个是主体的单位tensor：用于存放的是实实在在的这个自变量x 的数值
        # 另外一个是辅助的单位tensor：用于存放的是这个自变量x 被求导之后的导函数值
        # 通过一些正常手段（例如使用Tensor类的函数）进行操作的时候都是操作在主体那个单位tensor上，
        # 剩下那个辅助的单位tensor，我们可以通过x.grad来访问那个辅助的单位tensor。

        # 如果自变量x 不经过导数计算，x.grad是不会有数据的，你访问得到是None。
        # 所以f.backward()所做的就是把导数值存放在x.grad里面。
        #
        # 需要注意的是x.grad不会自动清零的，他只会不断把新得到的数值累加到旧的数值上面，
        # 这就需要我们利用optimizer.zero_grad()来给他清零。

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

    def save_models(self, episode_cnt, folder):
        """
        saves the target actor and critic models
        :param folder:
        :param episode_cnt: the count of episodes iterated
        :return:
        """
        # torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的weight和bias系数
        torch.save(self.target_actor.state_dict(), folder + '/' + str(episode_cnt) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), folder + '/' + str(episode_cnt) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode, folder):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param folder:
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(folder + '/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(folder + '/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')
