import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def lp_symmetry_shift(core, one_electron, two_electron, n_elec):
    """
    Performs a global Linear Programming optimization (LP-BLISS) directly on 
    molecular integral tensors to minimize the 1-norm prior to CDF factorization.
    
    Args:
        core (float): Nuclear repulsion / core scalar energy.
        one_electron (np.ndarray): 1-body integral tensor T_pq (N x N).
        two_electron (np.ndarray): 2-body integral tensor in chemist notation V_pqrs (N x N x N x N).
        n_elec (int): Number of active electrons.
        
    Returns:
        tuple: (core_shift, one_shift, two_shift) optimized tensors matching PennyLane shapes.
    """
    try:
        N = one_electron.shape[0]
        
        # Optimization variables:
        # c1 (scalar shift for N), c2 (scalar shift for N^2), and symmetric matrix xi_ij (for T_ij * N)
        xi_indices = [(i, j) for i in range(N) for j in range(i, N)]
        num_xi = len(xi_indices)
        num_vars = 2 + num_xi  # c1, c2, and unique xi_ij
        
        # Build index mapping for xi
        xi_map = {}
        for idx, (i, j) in enumerate(xi_indices):
            xi_map[(i, j)] = 2 + idx
            xi_map[(j, i)] = 2 + idx
            
        core_val = float(np.ravel(core)[0])
        one_flat = np.ravel(one_electron)
        two_flat = np.ravel(two_electron)
        orig_elements = np.concatenate([[core_val], one_flat, two_flat])
        M = len(orig_elements)
        
        # S_gradient maps how each parameter modifies elements of [core, T_pq, V_pqrs]
        S_gradient = np.zeros((num_vars, M))
        
        # 1. c1 shift: -c1 * (N_e - n_elec)
        # Adds +c1 * n_elec to core, and -c1 to 1-body diagonal T_ii
        S_gradient[0, 0] = float(n_elec)
        for i in range(N):
            one_idx = 1 + i * N + i
            S_gradient[0, one_idx] -= 1.0
            
        # 2. c2 shift: -c2 * (N_e^2 - n_elec^2)
        # Adds +c2 * n_elec^2 to core, -c2 to 1-body diagonal T_ii, -2*c2 to 2-body V_iijj
        S_gradient[1, 0] = float(n_elec ** 2)
        for i in range(N):
            one_idx = 1 + i * N + i
            S_gradient[1, one_idx] -= 1.0
            for j in range(N):
                two_idx = 1 + N*N + i*(N*N*N) + i*(N*N) + j*N + j
                S_gradient[1, two_idx] -= 2.0
                
        # 3. xi_ij shift: -sum_{ij} xi_ij (T_ij + T_ji) * (N_e - n_elec)
        for i, j in xi_indices:
            var_idx = xi_map[(i, j)]
            # Adds xi_ij * n_elec to T_ij and T_ji
            one_ij = 1 + i * N + j
            one_ji = 1 + j * N + i
            S_gradient[var_idx, one_ij] += float(n_elec)
            if i != j:
                S_gradient[var_idx, one_ji] += float(n_elec)
            # Subtracts xi_ij from 2-body terms V_ijkk
            for k in range(N):
                two_ijkk = 1 + N*N + i*(N*N*N) + j*(N*N) + k*N + k
                S_gradient[var_idx, two_ijkk] -= 1.0
                if i != j:
                    two_jikk = 1 + N*N + j*(N*N*N) + i*(N*N) + k*N + k
                    S_gradient[var_idx, two_jikk] -= 1.0

        # Construct LP L1-norm formulation
        c_lp = np.concatenate([np.zeros(num_vars), np.ones(M)])
        
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(M):
            # Upper bound constraint: S_gradient * vars - slack <= -orig_elements
            for j in range(num_vars):
                if S_gradient[j, i] != 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(S_gradient[j, i])
            row_indices.append(i)
            col_indices.append(num_vars + i)
            data.append(-1.0)
            
            # Lower bound constraint: -S_gradient * vars - slack <= orig_elements
            for j in range(num_vars):
                if S_gradient[j, i] != 0:
                    row_indices.append(M + i)
                    col_indices.append(j)
                    data.append(-S_gradient[j, i])
            row_indices.append(M + i)
            col_indices.append(num_vars + i)
            data.append(-1.0)
            
        A_ub = coo_matrix((data, (row_indices, col_indices)), shape=(2 * M, num_vars + M))
        b_ub = np.concatenate([-orig_elements, orig_elements])
        
        bounds = [(None, None)] * num_vars + [(0.0, None)] * M
        
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            print(f"LP-BLISS warning: HiGHS solver failed ({res.message}). Falling back to PennyLane symmetry_shift.")
            import pennylane as qml
            return qml.qchem.symmetry_shift(core, one_electron, two_electron, n_elec=n_elec)
            
        opt_vars = res.x[:num_vars]
        c1_opt, c2_opt = opt_vars[0], opt_vars[1]
        
        xi_opt = np.zeros((N, N))
        for (i, j), var_idx in xi_map.items():
            xi_opt[i, j] = opt_vars[var_idx]
            
        # Reconstruct shifted tensors
        core_shift = [core_val + c1_opt * n_elec + c2_opt * (n_elec ** 2)]
        one_shift = np.copy(one_electron)
        two_shift = np.copy(two_electron)
        
        for i in range(N):
            one_shift[i, i] -= (c1_opt + c2_opt)
            for j in range(N):
                two_shift[i, i, j, j] -= 2.0 * c2_opt
                
        for i in range(N):
            for j in range(N):
                one_shift[i, j] += xi_opt[i, j] * n_elec
                for k in range(N):
                    two_shift[i, j, k, k] -= xi_opt[i, j]
                    
        return core_shift, one_shift, two_shift

    except Exception as e:
        print(f"LP-BLISS exception ({e}). Falling back to PennyLane symmetry_shift.")
        import pennylane as qml
        return qml.qchem.symmetry_shift(core, one_electron, two_electron, n_elec=n_elec)
