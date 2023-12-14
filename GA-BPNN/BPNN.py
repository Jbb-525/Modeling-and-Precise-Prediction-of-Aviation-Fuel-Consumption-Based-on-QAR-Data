import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import functools
import torch.utils.data as Data
torch.set_default_tensor_type(torch.DoubleTensor)


class OneHiddenLayerNet(nn.Module):
    """  单隐藏层  """

    def __init__(self, input_size, hidden_size, output_size):
        super(OneHiddenLayerNet, self).__init__()

        self.onehiddenlayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
#             nn.Sigmoid()
        )

        # self.linear1 = torch.nn.Linear(input_size, hidden_size)
        # self.relu1 = torch.nn.Tanh()
        # self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_pred = self.onehiddenlayernet(x)

        # Z1 = self.linear1(x)
        # A1 = self.relu1(Z1)
        # y_pred = self.linear2(A1)

        return y_pred


class TwoHiddenLayerNet(nn.Module):
    """ 双隐藏层；单隐藏层最佳隐藏节点数用于第二个hiddenlayer，调第一个hiddenlayer节点数"""

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(TwoHiddenLayerNet, self).__init__()
        self.twohiddenlayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Sigmoid(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        y_pred = self.twohiddenlayernet(x)

        return y_pred


class ScheduledOptim(object):
    """  learning rate scheduling 
    Open the learning rate adjustment interface in optimizer to inherit the corresponding optimizer  """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr


import functools
def levenberg_marquardt(model,device, x_train, y_train, alpha):
    epoch = 1000
    loss_recorder = []

    for i in range(epoch):
        model.train()
        y_pred = model(x_train)

        Loss = nn.MSELoss(reduction='mean')
        loss = Loss(y_pred, y_train)

        prev_loss = loss.item()

        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        if_first = 0
        for g in gradients:
            g_vector = g.contiguous().view(-1) if if_first == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            if_first = 1

        l = g_vector.size(0)
        hessian = torch.zeros(l, l).to(device)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            if_first = 0
            for g in grad2rd:
                g_2 = g.contiguous().view(-1) if if_first == 0 else torch.cat([g_2, g.contiguous().view(-1)])
                if_first = 1
            hessian[idx] = g_2

            # 使用jacobian.T*jacobian近似替代hessian
#         g_vector.unsqueeze_(0).to(device)
#         hessian = torch.matmul(g_vector.T,g_vector)

        g_vector.unsqueeze_(0).to(device)
        dx = -1 * (alpha * torch.eye(hessian.shape[-1]).to(device) + hessian).inverse().mm(g_vector.T).detach()

        cnt = 0  # 更新参数的位置
        model.zero_grad()

        for p in model.parameters():
            p_shape = torch.Tensor([p.shape]).tolist()[0]
            num = int(functools.reduce(lambda x, y: x * y, p_shape, 1))  # 此次更新参数的个数
            p.requires_grad = False
            p += dx[cnt:cnt + num].reshape(p.shape)
            cnt += num
            p.requires_grad = True

        y_pred = model(x_train)
        loss_ = Loss(y_pred, y_train)
        loss_out = loss_.item()

        if loss_out < prev_loss:  # 迭代成功
                # print("Successful iteration")
            loss_recorder.append(loss_out)
            alpha /= 10
        else:  # 增加步长
                # print("Augmenting step size")
            alpha *= 10
            cnt = 0
            for p in model.parameters():
                p_shape = torch.Tensor([p.shape]).tolist()[0]
                num = int(functools.reduce(lambda x, y: x * y, p_shape, 1))
                p.requires_grad = False
                p -= dx[cnt:cnt + num, :].reshape(p.shape)
                cnt += num
                p.requires_grad = True
                    

    return np.mean(loss_recorder), loss_recorder


def adjusting_lr(x, y, model):


    """ 从一个较低的学习率开始训练一个网络，并以指数级增长每一批的学习率 """
    basic_optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    optimizer = ScheduledOptim(basic_optim)
    lr_mult = (1 / 1e-5) ** (1 / 10000)
    lr = []
    losses = []
    cv_losses = []
    best_loss = 1e9
    EPOCH = 10000

    from sklearn.model_selection import train_test_split
    x_train_, x_cv, y_train_, y_cv = train_test_split(x, y, test_size=0.2)

    for t in range(EPOCH):
        out = model(x)
        Loss = nn.MSELoss(reduction='mean')
        loss = Loss(out, y)

        # cv_out = model(x_cv)
        # cv_loss = Loss(cv_out, y_cv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (optimizer.learning_rate >= 1e-2) & (optimizer.learning_rate <= 1e-1):
        #     lr.append(optimizer.learning_rate)
        #     losses.append(loss.data)
        #     # cv_losses.append(cv_loss.data)

        lr.append(optimizer.learning_rate)
        losses.append(loss.item())

        optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
        #         if loss.data < best_loss:
        #             best_loss = loss.data
        if optimizer.learning_rate > 0.5:
            break

    plt.figure()
    plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
#     print(losses)
    plt.plot(np.log(lr), losses, label='train')
    # plt.plot(lr, cv_losses, label='cv')
    plt.legend()
    plt.show()


def train_model_lrfix(x,y,model,learning_rate,m):
    """ 学习率固定 """
    """
    x---训练集数据
    y---训练集标签
    model---模型
    learning——rate---学习率
    m---动量率
    """
    Loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,  momentum=m)

    loss_recorder = []
    EPOCH = 100000

    for t in range(EPOCH):
        y_pred = model(x)
        loss = Loss(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_recorder.append(loss.item())

    return np.mean(loss_recorder), loss_recorder


def train_model_lrvar(x,y,model,learning_rate,m):
    """ 学习率衰退 """
    """
    x---训练集数据
    y---训练集标签
    model---模型
    learning——rate---学习率
    """
    Loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=m)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    loss_recorder = []
    EPOCH = 10000

    for t in range(EPOCH):
        y_pred = model(x)
        loss = Loss(y_pred, y)
        loss_recorder.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(loss_recorder), loss_recorder
