import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
    # fani: 指定值 或者 第一维大小
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    # pytorch网络结构定义实例：https://www.jb51.net/article/178460.htm
    def __init__(self, s_dim: int, a_dim: int):
        super(Critic, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        # state 网络
        # 全连接层
        # nn.Linear(in_feature,out_feature,bias)
        # in_feature: 输入Tensor最后一维的通道数
        # out_feature: 返回Tensor最后一维的通道数
        self.fcs1 = nn.Linear(s_dim, 256)
        # .weight.data：得到的是一个Tensor的张量（向量），不可训练的类型
        # .weight：得到的是一个parameter的变量，可以计算梯度的
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        # action 网络
        self.fca1 = nn.Linear(a_dim, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 1)
        # 设置fc3 Linear全连接层的权重 是（-EPS, EPS）之间的均匀分布
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, s, a):
        """
        returns Value function Q(s,a) obtained from critic network
        :param s: Input state (Torch Variable : [n,state_dim] )
        :param a: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        # n: n 个 样本

        # F.relu()是函数调用，一般使用在forward函数里。
        # 激活函数的作用：增加非线性因素，解决线性模型表达能力不足的缺陷。
        # 没有激活函数的神经网络实质上是一个线性回归模型，只能解决线性可分的问题.
        # s：n * s_dim，经过fcs1，s1：n * 256
        s1 = F.relu(self.fcs1(s))
        # s1：n * 256，经过fcs2，s2：n * 128
        s2 = F.relu(self.fcs2(s1))
        # 上述 在对 状态张量 维度处理，由 s_dim -> 256 -> 128

        # a：n * a_dim，经过fca1，a1：n * 128
        a1 = F.relu(self.fca1(a))

        # torch.cat(): 在给定维度上对输入的张量序列seq 进行连接操作。
        # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
        # dim : 选择的扩维, 必须在0到len(inputs[0].shape)之间，沿着此维连接张量序列。
        # dim = 1，若两个 2 * 3的张量，则是 变成一个 2 * 6，并排拼接，列数增加

        # 拼接后，x: n * 256
        x = torch.cat((s2, a1), dim=1)
        # 开始 计算 Q(s, a)
        # x：n * 256，经过fc2，x：n * 128
        x = F.relu(self.fc2(x))
        # x：n * 128，经过fc3，x：n * 1
        x = self.fc3(x)
        # n 条经验 计算的 Q值
        return x


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_lim):
        """
        :param s_dim: Dimension of input state (int)
        :param a_dim: Dimension of output action (int)
        :param a_lim: Used to limit action in [-action_lim, action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_lim = a_lim

        self.fc1 = nn.Linear(s_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64, a_dim)
        self.fc4.weight.data.uniform_(-EPS, EPS)

    def forward(self, s):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param s: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        a = F.tanh(self.fc4(x))
        # a_lim: Used to limit action in [-action_lim, action_lim]
        a = a * self.a_lim

        return a