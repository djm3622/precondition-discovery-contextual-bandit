from scipy.sparse.linalg import spilu, LinearOperator
from scipy.sparse.linalg import spilu
from pyamg import smoothed_aggregation_solver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def jacobi_preconditioner(A):
    M = np.eye(A.shape[0]) * (1 / np.diag(A))
    return M


def ilu_preconditioner(A, fill_factor=1):
    ilu = spla.spilu(A, fill_factor)
    Mx = lambda x: ilu.solve(x)
    M = spla.LinearOperator(A.shape, Mx)
    return M


def amg_preconditioner(A, max_levels=10, strength='classical'):
    ml = smoothed_aggregation_solver(A, max_levels=max_levels, strength=strength)
    M = LinearOperator(A.shape, matvec=ml.solve, rmatvec=ml.solve)
    return M


def get_traditional_preconditioners(batch_matrix, ind, fill_factor, max_levels, strength):
    mat = batch_matrix[ind].astype(np.float64)
    jac = jacobi_preconditioner(mat)
    ilu = ilu_preconditioner(mat, fill_factor)
    amg = amg_preconditioner(mat, max_levels, strength)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    im1 = ax1.matshow(jac)
    ax1.set_title(f'Jacobi')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    im2 = ax2.matshow(ilu @ np.eye(mat.shape[0]))
    ax2.set_title(f'ILU({fill_factor})')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    im3 = ax3.matshow(amg @ np.eye(mat.shape[0]))
    ax3.set_title(f'AMG({10})')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    return jac, ilu, amg


def check_conditioning(batch_mat, M, ind):
    mat = batch_mat[ind]
    inner = M @ mat
    cond = np.mean(np.linalg.norm(inner, ord="fro") * np.linalg.norm(np.linalg.inv(inner), ord="fro"))
    print(f'Condition: {cond}')