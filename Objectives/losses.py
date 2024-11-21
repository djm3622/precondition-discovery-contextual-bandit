from torch import nn
import torch


class CondEyeDistance(nn.Module):
    def __init__(self, l1=1e-6, inv=1e-5, dev=1.0, cond=1e-2, batch_size=64, size=25):
        super().__init__()
        self.l1 = l1  # l1 norm of preconditioner
        self.inv = inv  # computes distance of inner product from idenity (collapses to 0s if weighted too large)
        self.dev = dev  # measure the signular values deviation from 1s
        self.cond = cond  # measuer of logged condition values
        self.size = size
        self.batch_size = batch_size
        
    def forward(self, inp, outp):
        avg_loss = 0
        inner_product = torch.bmm(outp, inp)
        identity = torch.eye(self.size).expand(self.batch_size, self.size, self.size).to(outp.device)
                
        # minimize distance from inner product to the idenityt
        avg_loss += torch.mean(torch.norm(inner_product - identity, p='fro', dim=(1, 2))) * self.inv

        _, S, _ = torch.linalg.svd(inner_product)
        # svd values to be close to 1, low conditioning number
        avg_loss += torch.mean((S - 1)**2) * self.dev 
        avg_loss += torch.mean(torch.log(S.max(dim=-1).values) - torch.log(S.min(dim=-1).values)) * self.cond 

        # optional regularization
        if self.l1 is not None:
            avg_loss += self.l1 * torch.norm(outp, p=1)
        
        return avg_loss