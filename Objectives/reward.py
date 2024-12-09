import torch
from torch_cg.torch_cg import cg_batch


# Compute the the ground truth for the batched reward
class batched_reward():
    def __init__(self, size, verbose, switch_up):
        self.size = size
        self.verbose = verbose
        self.switch_up = switch_up
    
    def forward(self, A, M, b):
        _, info = cg_batch(
            lambda x: torch.matmul(A, x), b,                   
            lambda x: torch.matmul(M, x), 
            maxiter=self.size, 
            verbose=self.verbose
        )
        niter, optimal, last_residual = info['niter'], info['optimal'], info['last_residual']
        
        sparse_v = -((self.size**2) - torch.count_nonzero(M, dim=(1, 2))) if self.switch_up else 1 - (torch.count_nonzero(M, dim=(1, 2)) / (self.size**2)) # 1 - (non / total) = % of zero
        iters_v = -niter if self.switch_up else (0 if not optimal else 1/niter)
        res_v = -last_residual if self.switch_up else 1/(1+last_residual)
                
        return sparse_v, iters_v, res_v
    
    
class mod_batched_reward():
    def __init__(self, size, verbose, switch_up):
        self.size = size
        self.verbose = verbose
        self.switch_up = switch_up
    
    def forward(self, A, M, b):
        _, info = cg_batch(
            lambda x: torch.matmul(A, x), b,                   
            lambda x: torch.matmul(M, x), 
            maxiter=self.size, 
            verbose=self.verbose
        )
        niter, optimal, last_residual = info['niter'], info['optimal'], info['last_residual']
        
        sparse_v = -((self.size**2) - torch.count_nonzero(M, dim=(1, 2))) if self.switch_up else 1 - (torch.count_nonzero(M, dim=(1, 2)) / (self.size**2)) # 1 - (non / total) = % of zero
        res_v = -last_residual if self.switch_up else 1/(1+last_residual)
        iters_v = -niter if self.switch_up else (0 if not optimal else 1/niter)
        # modified part
        _, S, _ = torch.linalg.svd(torch.bmm(A, M))
        # svd values to be close to 1, low conditioning number
        cond = 1/(torch.log(S.max(dim=-1).values) - torch.log(S.min(dim=-1).values))

        return sparse_v, res_v, iters_v, cond