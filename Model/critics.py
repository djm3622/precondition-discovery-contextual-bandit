import torch
from torch import nn


class SingleCritic(nn.Module):
    def __init__(self, n=25, down=256, batch_size=64, sigmoid_scale=False):
        super().__init__()
        
        self.input_dim = 2 * (n**2) # both state and action are concatonated before input
        self.batch_size = batch_size
        self.sigmoid_scale = sigmoid_scale # passes output through sigmoid and then scales by 3 to be in range [0, 3]
        
        self.fcn1 = nn.Linear(self.input_dim, self.input_dim-down)
        self.fcn2 = nn.Linear(self.input_dim-down, self.input_dim-2*down)
        self.fcn3 = nn.Linear(self.input_dim-2*down, self.input_dim-3*down)
        self.fcn4 = nn.Linear(self.input_dim-3*down, self.input_dim-4*down)
        self.fcn5 = nn.Linear(self.input_dim-4*down, 1)
        
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.act(self.fcn2(out))        
        out = self.act(self.fcn3(out))        
        out = self.act(self.fcn4(out))
        out = self.fcn5(out)
        
        if self.sigmoid_scale:
            out = 3 * self.act_out(out)
            
        return out
    

class SmallerSingleCritic(nn.Module):
    def __init__(self, n=25, down=256, batch_size=64, sigmoid_scale=False, scale=3):
        super().__init__()
        
        self.input_dim = 2 * (n**2) # both state and action are concatonated before input
        self.batch_size = batch_size
        self.sigmoid_scale = sigmoid_scale # passes output through sigmoid and then scales by 3 to be in range [0, 3]
        self.scale = scale
        
        self.fcn1 = nn.Linear(self.input_dim, self.input_dim-512)
        self.fcn2 = nn.Linear(self.input_dim-512, self.input_dim-512-256)
        self.fcn = nn.Linear(self.input_dim-512-256, 1)
        
        self.act = nn.Tanh()
        self.act_out = nn.Sigmoid()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.act(self.fcn2(out))
        out = self.fcn(out)
        
        if self.sigmoid_scale:
            out = self.scale * self.act_out(out)
            
        return out
    
    
class MultiCritic(nn.Module):
    def __init__(self, n=25, down=256, batch_size=64, sigmoid_scale=False):
        super().__init__()
        
        self.input_dim = 2 * (n**2) # both state and action are concatonated before input
        self.batch_size = batch_size
        self.sigmoid_scale = sigmoid_scale # passes output through sigmoid and then scales by 3 to be in range [0, 3]
        
        self.fcn1 = nn.Linear(self.input_dim, self.input_dim-down)
        self.fcn2 = nn.Linear(self.input_dim-down, self.input_dim-2*down)
        self.fcn3 = nn.Linear(self.input_dim-2*down, self.input_dim-3*down)
        self.fcn4 = nn.Linear(self.input_dim-3*down, self.input_dim-4*down)
        self.fcn5 = nn.Linear(self.input_dim-4*down, 3)
        
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.act(self.fcn2(out))        
        out = self.act(self.fcn3(out))        
        out = self.act(self.fcn4(out))
        out = self.fcn5(out)
        
        if self.sigmoid_scale:
            out = self.act_out(out)
            
        return out
    
    
class SmallerMultiCritic(nn.Module):
    def __init__(self, n=25, down=256, batch_size=64, sigmoid_scale=False):
        super().__init__()
        
        self.input_dim = 2 * (n**2) # both state and action are concatonated before input
        self.batch_size = batch_size
        self.sigmoid_scale = sigmoid_scale # passes output through sigmoid and then scales by 3 to be in range [0, 3]
        
        self.fcn1 = nn.Linear(self.input_dim, self.input_dim-512)
        self.fcn = nn.Linear(self.input_dim-512, 3)
        
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.fcn(out)
        
        if self.sigmoid_scale:
            out = self.act_out(out)
            
        return out
        
        
def critic_step():
    pass