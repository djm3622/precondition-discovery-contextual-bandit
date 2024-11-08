import numpy as np
import torch
import matplotlib.pyplot as plt


def view_matrix(batch_mat, ind):
    plt.matshow(batch_mat[ind])

    
def check_singularity(batch_mat, ind):
    mat = batch_mat[ind]
    det_mat = np.linalg.det(mat)
    print(f'Determinent: {det_mat}')

    if np.isclose(det_mat, 0):
        print("Matrix is singular.")
        return False
        
    rank_mat = np.linalg.matrix_rank(mat)
    if rank_mat < mat.shape[0]:
        print("Matrix is singular.")
        return False
    
    print("Matrix is non-singular.")
        
    return True


def check_conditioning(batch_mat, ind):
    mat = batch_mat[ind]
    _, S, _ = torch.linalg.svd(mat)
    cond = S.max() / S.min()
    print(f'Condition: {cond}')


def normalize_features(graph_data):
    x_mean = graph_data.mean()
    x_std = graph_data.std()
    out = (graph_data - x_mean) / (x_std)
    return out.float()


def set_device(device_pref, ind_dev=0):
    device = None
    
    if device_pref == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{ind_dev}')
        print('Now using GPU.')
    else:
        device = torch.device('cpu')
        if device_pref == 'cuda':
            print('GPU not available, defaulting to CPU.')
        else:
            print('Now using CPU.')
    
    return device


def seed(seed):
    torch.manual_seed(seed)