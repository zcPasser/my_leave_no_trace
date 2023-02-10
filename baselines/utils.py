import numpy as np
import torch
import shutil
import torch.autograd as Variable
from models import Actor, Critic

def create_agent(s_dim, a_dim, a_lim):
    actor = Actor(s_dim=s_dim, a_dim=a_dim, a_lim=a_lim)
    target_actor = Actor(s_dim=s_dim, a_dim=a_dim, a_lim=a_lim)

    critic = Critic(s_dim=s_dim, a_dim=a_dim)
    target_critic = Critic(s_dim=s_dim, a_dim=a_dim)

    hard_update(target_actor, actor)
    hard_update(target_critic, critic)

    return {
        'actor': actor,
        'target_actor': target_actor,
        'critic': critic,
        'target_critic': target_critic,
        's_dim': s_dim,
        'a_dim': a_dim,
        'a_lim': a_lim
    }

def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            param.data
        )


def save_training_checkpoint(s, is_best, episode_cnt):
    """
    Saves the models, with all training parameters intact
    :param s:
    :param is_best:
    :param episode_cnt:
    :return:
    """
    file_name = str(episode_cnt) + 'checkpoint.path.rar'
    # 保存一个序列化（serialized）的目标到磁盘。
    # 函数使用了Python的pickle程序用于序列化。
    # 模型（models），张量（tensors）和文件夹（dictionaries）都是可以用这个函数保存的目标类型。
    torch.save(s, file_name)
    if is_best:
        # shutil.copyfile(src, dst):
        # 将名为src的文件的内容（无元数据）复制到名为dst的文件中 。
        # dst必须是完整的目标文件名
        shutil.copyfile(file_name, 'model_best.pth.tar')

# 添加OU过程噪声，提高 探索效率。
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, a_dim, mu=0, theta=0.15, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.a_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.a_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
