from torch import nn
import torch


class CondEyeDistance(nn.Module):
    def __init__(self, l1=1e-6, inv=1e-5, dev=1.0, cond=1e-2, batch_size=64, size=25):
        super().__init__()
        self.l1 = l1
        self.inv = inv
        self.dev = dev
        self.cond = cond
        self.size = size
        self.batch_size = batch_size
        
    def forward(self, inp, outp):
        avg_loss = 0
        inner_product = torch.bmm(outp, inp)
        identity = torch.eye(self.size).expand(self.batch_size, self.size, self.size).to(outp.device)
                
        avg_loss += torch.mean(torch.norm(inner_product - identity, p='fro', dim=(1, 2))) * self.inv

        _, S, _ = torch.linalg.svd(inner_product)
        deviation_from_one = torch.mean((S - 1)**2)  
        avg_loss += deviation_from_one * self.dev
        avg_loss += (torch.log(S.max()) - torch.log(S.min())) * self.cond

        if self.l1 is not None:
            avg_loss += self.l1 * torch.norm(outp, p=1)
        
        return avg_loss