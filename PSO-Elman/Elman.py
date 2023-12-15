import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

class Elman(nn.Module):
    def __init__(self,hidden_size,epochs,lr):
        super(Elman, self).__init__()
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        
    def forward(self,input_, context_state, w1, w2, w3, b1, b2):
        hidden_layer = torch.tanh(input_.mm(w1) + b1+context_state.mm(w3))
        out = torch.sigmoid(hidden_layer.mm(w2) + b2)
        context_state = torch.sigmoid(input_.mm(w1) + b1)
        return  (out, context_state)
    
    
    def train(self,x_train,y_train,w1,w2,w3,b1,b2):
        Loss = torch.nn.MSELoss(reduction='mean')
        total_loss=[]
        for i in range(self.epochs):
            y_pred = []
            context_state = Variable(torch.zeros((1, self.hidden_size)).type(torch.DoubleTensor), requires_grad=False)
            for j in range(x_train.size(0)):
                input_ = x_train[j:j+1]
                target = y_train[j:j+1]
                (pred, context_state) = self.forward(input_, context_state, w1,w2,w3,b1,b2)
                y_pred.append(pred.item())
                loss = Loss(pred, target)
                loss.requires_grad_(True)
                loss.backward()
                w1.data -= self.lr * (w1.grad.data)
                w2.data -=  self.lr * (w2.grad.data)
                w3.data -=  self.lr * (w3.grad.data)
                b1.data -=  self.lr * (b1.grad.data)
                b2.data -=  self.lr * (b2.grad.data)
                w1.grad.data.zero_()
                w2.grad.data.zero_()
                w3.grad.data.zero_()
                b1.grad.data.zero_()
                b2.grad.data.zero_()
                context_state = Variable(context_state.data)
                
            loss = Loss(torch.tensor(y_pred).reshape(y_train.shape),y_train)
            total_loss.append(loss.item())
        return total_loss,(w1,w2,w3,b1,b2)
    
    def test(self,x_test,y_test,w1,w2,w3,b1,b2):
        y_hat = []
        for i in range(x_test.size(0)):
            input_ = x_test[i:i+1]
            (pred, context_state) = forward(input_, context_state, w1,w2,w3,b1,b2)
            context_state = context_state
            y_hat.append(pred.item())
        
        loss = Loss(y_hat,y_test[0:-1]).item()
        
        return loss, y_hat