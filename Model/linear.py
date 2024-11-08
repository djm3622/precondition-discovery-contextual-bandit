import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, n=25, hidden=256, batch_size=64, sparse_tol=1e-5):
        super().__init__()
        
        self.n = n
        self.batch_size = batch_size
        self.sparse_tol = sparse_tol
        lower_triangle_size = (n * (n + 1)) // 2
        
        self.fcn1 = nn.Linear(n * n, hidden)
        self.fcn2 = nn.Linear(hidden, hidden)
        self.fcn3 = nn.Linear(hidden, hidden)
        self.fcn4 = nn.Linear(hidden, hidden)
        self.fcn5 = nn.Linear(hidden, hidden)
        self.fcn6 = nn.Linear(hidden, lower_triangle_size)
        
        self.act = nn.ReLU()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.act(self.fcn2(out))        
        out = self.act(self.fcn3(out))        
        out = self.act(self.fcn4(out))
        out = self.act(self.fcn5(out))
        
        # Output the lower triangular part
        lower_tri_values = self.fcn6(out)
        
        # Initialize a full zero matrix
        full_matrix = torch.zeros((self.batch_size, self.n, self.n), device=lower_tri_values.device)
        
        # Fill in the lower triangular part of the matrix
        tril_indices = torch.tril_indices(self.n, self.n)
        full_matrix[:, tril_indices[0], tril_indices[1]] = lower_tri_values
        
        # Use symmetry to fill the upper triangular part
        diags = torch.diagonal(full_matrix, dim1=1, dim2=2).unsqueeze(-1) * torch.eye(self.n).to(full_matrix.device)
        full_matrix = full_matrix + torch.transpose(full_matrix, dim0=2, dim1=1) - diags
        
        return torch.where(torch.abs(full_matrix) < self.sparse_tol, torch.zeros_like(full_matrix), full_matrix)