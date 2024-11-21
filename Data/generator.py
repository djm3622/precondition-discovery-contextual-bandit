import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def generate_2d_diffusion_spd(n, bump_range=(3, 10), contrast_range=(0.1, 5.0), perturbation_strength=1e-4):
    L = 1.0
    h = L / (n - 1)
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    
    # Random number of bumps
    num_bumps = np.random.randint(bump_range[0], bump_range[1])
    
    # Generate random 2D diffusion coefficient a(x,y)
    a = np.ones_like(X)
    for _ in range(num_bumps):
        center_x = np.random.uniform(0, L)
        center_y = np.random.uniform(0, L)
        width = np.random.uniform(0.03, 0.3)
        height = np.random.uniform(contrast_range[0], contrast_range[1])
        a += height * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * width**2))
    
    # Build sparse matrix for 2D problem
    N = n * n  # total size
    main_diag = np.ones(N) * 1e-1 
    off_diag_x = np.zeros(N - 1)
    off_diag_y = np.zeros(N - n)
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            
            if j < n - 1:  # x-direction neighbor
                a_mid = (a[i, j] + a[i, j + 1]) / 2
                off_diag_x[idx] = -a_mid / h**2
                main_diag[idx] += a_mid / h**2
                main_diag[idx + 1] += a_mid / h**2
                
            if i < n - 1:  # y-direction neighbor
                a_mid = (a[i, j] + a[i + 1, j]) / 2
                off_diag_y[idx] = -a_mid / h**2
                main_diag[idx] += a_mid / h**2
                main_diag[idx + n] += a_mid / h**2
    
    # Construct sparse matrix
    diagonals = [off_diag_y, off_diag_x, main_diag, off_diag_x, off_diag_y]
    offsets = [-n, -1, 0, 1, n]
    A = sp.diags(diagonals, offsets, format="csr")
    
    A = perturb_nonzeros(A, perturbation_strength)
    
    return A, a

def perturb_nonzeros(A, perturbation_strength):
    # Ensure A is in CSR format for efficient indexing
    A = A.tocsr()

    # Extract nonzero elements (row, col, and values)
    rows, cols = A.nonzero()
    values = A.data

    # Generate random perturbations for the nonzero values
    perturbations = np.random.uniform(-perturbation_strength, perturbation_strength, size=values.shape)
    
    # Apply perturbation to the nonzero values
    A.data += perturbations

    # Ensure symmetry by adding the transpose of the perturbed matrix
    A = (A + A.T) / 2

    return A


def generate_random_helmholtz(n, density=0.1):
    L = 1.0
    k = np.random.uniform(1, 2)  # Random wavenumber
    h = L / (n - 1)  

    # Discretization of Helmholtz operator (1D)
    diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]

    helmholtz = sp.diags(diagonals, [-1, 0, 1]) / h**2 + k**2 * sp.eye(n)

    # Ensure no perturbations on the tridiagnonal
    perturb = sp.random(n, n, density=density) * np.max(helmholtz)
    
    perturb.setdiag(0)  # Main diagonal
    perturb.setdiag(0, k=1)  # First upper diagonal
    perturb.setdiag(0, k=-1)  # First lower diagonal
        
    return helmholtz+perturb