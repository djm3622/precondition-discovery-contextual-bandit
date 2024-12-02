import torch
from torch_cg.torch_cg import cg_batch


# Compute the the ground truth for the batched reward
class batched_reward():
    def __init__(self, size, verbose):
        self.size = size
        self.verbose = verbose
    
    def forward(self, A, M, b):
        _, info = cg_batch(
            lambda x: torch.matmul(A, x), b,                   
            lambda x: torch.matmul(M, x), 
            maxiter=self.size, 
            verbose=self.verbose
        )
        niter, optimal, last_residual = info['niter'], info['optimal'], info['last_residual']
        
        sparse_v = 1 - (torch.count_nonzero(M, dim=(1, 2)) / (self.size**2)) # 1 - (non / total) = % of zero
        iters_v = 0 if not optimal else 1/niter
        res_v = 1/(1+last_residual)
                
        return sparse_v, iters_v, res_v