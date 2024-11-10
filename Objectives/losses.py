from torch import nn
import torch


class CondEyeDistance(nn.Module):
    def __init__(self, l1=1e-6, inv=1e-5, dev=1.0, cond=1e-2, lambda_diag=1e-2, batch_size=64, size=25):
        super().__init__()
        self.l1 = l1  # l1 norm of preconditioner
        self.inv = inv  # computes distance of inner product from idenity (collapses to 0s if weighted too large)
        self.dev = dev  # measure the signular values deviation from 1s
        self.cond = cond  # measuer of logged condition values
        self.lambda_diag = lambda_diag # penalizes entries on the diagonal for being far max of matrix
        self.size = size
        self.batch_size = batch_size
        
    def forward(self, inp, outp):
        avg_loss = 0
        inner_product = torch.bmm(outp, inp)
        identity = torch.eye(self.size).expand(self.batch_size, self.size, self.size).to(outp.device)
                
        # minimize distance from inner product to the idenityt
        avg_loss += torch.mean(torch.norm(inner_product - identity, p='fro', dim=(1, 2))) * self.inv

        _, S, _ = torch.linalg.svd(inner_product)
        deviation_from_one = torch.mean((S - 1)**2) 
        # svd values to be close to 1, low conditioning number
        avg_loss += deviation_from_one * self.dev 
        avg_loss += torch.mean(torch.log(S.max(dim=-1).values) - torch.log(S.min(dim=-1).values)) * self.cond 
        
        # encourages a strong diagonal
        diag_loss = torch.mean(torch.norm(torch.amax(outp, dim=(1, 2)).unsqueeze(-1) - torch.diagonal(outp, dim1=1, dim2=2), p=2))
        avg_loss += diag_loss * self.lambda_diag

        # optional regularization
        if self.l1 is not None:
            avg_loss += self.l1 * torch.norm(outp, p=1)
        
        return avg_loss