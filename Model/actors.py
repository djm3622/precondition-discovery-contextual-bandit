import torch
import torch.nn as nn


class NonsymmetricIdFCN(nn.Module):
    def __init__(self, n=25, hidden=256, batch_size=64, sparse_tol=1e-5, diagonal_bias=None):
        super().__init__()
        
        self.n = n
        self.batch_size = batch_size
        self.sparse_tol = sparse_tol
        self.diagonal_bias = diagonal_bias
        
        self.fcn1 = nn.Linear(n * n, hidden)
        self.fcn2 = nn.Linear(hidden, hidden)
        self.fcn3 = nn.Linear(hidden, hidden)
        self.fcn4 = nn.Linear(hidden, hidden)
        self.fcn5 = nn.Linear(hidden, hidden)
        self.fcn6 = nn.Linear(hidden, n * n)
        
        self.act = nn.ReLU()
        
    def forward(self, inpt):
        out = self.act(self.fcn1(inpt))
        out = self.act(self.fcn2(out))        
        out = self.act(self.fcn3(out))        
        out = self.act(self.fcn4(out))
        out = self.act(self.fcn5(out))
        
        # Output the lower triangular part
        full_matrix = self.fcn6(out).view(self.batch_size, self.n, self.n)
        
        if self.diagonal_bias is not None:
            full_matrix = full_matrix + self.diagonal_bias * torch.eye(self.n).to(full_matrix.device)
        
        return torch.where(torch.abs(full_matrix) < self.sparse_tol, torch.zeros_like(full_matrix), full_matrix)

    
class SymmetricIdFCN(nn.Module):
    def __init__(self, n=25, hidden=256, batch_size=64, sparse_tol=1e-5, diagonal_bias=None):
        super().__init__()
        
        self.n = n
        self.batch_size = batch_size
        self.sparse_tol = sparse_tol
        self.diagonal_bias = diagonal_bias
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
        
        if self.diagonal_bias is not None:
            full_matrix = full_matrix + self.diagonal_bias * torch.eye(self.n).to(full_matrix.device)
        
        return torch.where(torch.abs(full_matrix) < self.sparse_tol, torch.zeros_like(full_matrix), full_matrix)
    
    
class CholeskyFCN(nn.Module):
    def __init__(self, n=25, hidden=256, batch_size=64, sparse_tol=1e-5, diagonal_bias=None, tanh=False):
        super().__init__()
        
        self.n = n
        self.batch_size = batch_size
        self.sparse_tol = sparse_tol
        self.diagonal_bias = diagonal_bias
        self.t = tanh
        lower_triangle_size = (n * (n + 1)) // 2
        
        self.fcn1 = nn.Linear(n * n, hidden)
        self.fcn2 = nn.Linear(hidden, hidden)
        self.fcn3 = nn.Linear(hidden, hidden)
        self.fcn4 = nn.Linear(hidden, hidden)
        self.fcn5 = nn.Linear(hidden, hidden)
        self.fcn6 = nn.Linear(hidden, lower_triangle_size)
        
        self.act = nn.ReLU()
        self.tanh = nn.Tanh()
        
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
        
        # compute choltsky
        full_matrix = full_matrix @ torch.transpose(full_matrix, dim0=2, dim1=1)
        
        if self.diagonal_bias is not None:
            full_matrix = full_matrix + self.diagonal_bias * torch.eye(self.n).to(full_matrix.device)
        
        out = torch.where(torch.abs(full_matrix) < self.sparse_tol, torch.zeros_like(full_matrix), full_matrix)
        return self.tanh(out) if self.t else out
    
    
class LuFCN(nn.Module):
    def __init__(self, n=25, hidden=128, batch_size=64, sparse_tol=1e-5, diagonal_bias=None):
        super().__init__()
        
        self.n = n
        self.batch_size = batch_size
        self.sparse_tol = sparse_tol
        self.diagonal_bias = diagonal_bias
        lower_triangle_size = (n * (n + 1)) // 2
        
        self.l_layers = []
        self.u_layers = []
        
        for _ in range(4):
            self.l_layers.append(nn.Linear(hidden, hidden))
            self.l_layers.append(nn.ReLU())
            self.u_layers.append(nn.Linear(hidden, hidden))
            self.u_layers.append(nn.ReLU())
            
        self.l_layers = nn.Sequential(*self.l_layers)
        self.u_layers = nn.Sequential(*self.u_layers)
        
        self.entry = nn.Linear(n * n, hidden)
        self.l_out = nn.Linear(hidden, lower_triangle_size)
        self.u_out = nn.Linear(hidden, lower_triangle_size)
        
        self.act = nn.ReLU()
        
    def forward(self, inpt):
        out = self.act(self.entry(inpt))
        lout = self.l_layers(out)
        uout = self.u_layers(out)  
        
        # Output the lower triangular part
        lower_tri_values = self.l_out(lout)
        # Output of upper triangular part
        upper_tri_values = self.u_out(uout)
        
        # Initialize a full zero matrix
        lower_full_matrix = torch.zeros((self.batch_size, self.n, self.n), device=lower_tri_values.device)
        upper_full_matrix = torch.zeros((self.batch_size, self.n, self.n), device=upper_tri_values.device)
        
        # Fill in the lower triangular part of the matrix
        tril_indices = torch.tril_indices(self.n, self.n)
        lower_full_matrix[:, tril_indices[0], tril_indices[1]] = lower_tri_values
        
        # Fill in the upper triangular part of the matrix
        triu_indices = torch.triu_indices(self.n, self.n)
        upper_full_matrix[:, triu_indices[0], triu_indices[1]] = upper_tri_values
        
        # compute LU
        full_matrix = lower_full_matrix @ upper_full_matrix
        
        if self.diagonal_bias is not None:
            full_matrix = full_matrix + self.diagonal_bias * torch.eye(self.n).to(full_matrix.device)
        
        return torch.where(torch.abs(full_matrix) < self.sparse_tol, torch.zeros_like(full_matrix), full_matrix)
    
    
    
def actor_step():
    pass