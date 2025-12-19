import torch
import numpy as np
import scipy.io as sio
import time
import os


device = torch.device('cpu')
print(f"Using device: {device}")

def standardization(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True)
    return (X - mean) / (std + 1e-12)

def gp(M, k):
    w = torch.sum(M * M, dim=1)  # (d,)
    _, ind = torch.sort(w, descending=True)
    W = torch.zeros_like(M)
    top_k_idx = ind[:k]
    W[top_k_idx, :] = M[top_k_idx, :]
    return W

def gen_initialization(k, d, c):
    W_init = torch.rand((d, c), device=device)
    W_init = gp(W_init, k)
    return W_init

def total_objective_new_yi(W, X, P):
    """
    Loss = || X - P * W^T * X ||_F^2     X: (d, n)     P: (d, c)    W: (d, c)
    """
    # term = X - P @ W.T @ X
    term = X - torch.mm(torch.mm(P, W.t()), X)
    loss = torch.norm(term, p='fro') ** 2
    return loss.item()

def solve_feature_selection_vectorized(A, B, V, VVT, index, k):
    # A: (d, d), B: (d, c), V: (k, c), VVT: (k, k)
    
    diag_A = torch.diagonal(A) # (d,)
    
    idx_tensor = torch.tensor(index, device=device, dtype=torch.long)
    
    for j in range(k):
        mask = torch.ones(k, dtype=torch.bool, device=device)
        mask[j] = False
        other_indices = idx_tensor[mask] # (k-1,)
        term_others = torch.mv(A[:, other_indices], VVT[mask, j]) # (d,)
        term_self = diag_A * VVT[j, j] # (d,)
        term_A_total = 2 * term_others + term_self
        term_B = 2 * torch.mv(B, V[j, :]) # (d,)
        obj_values = term_A_total - term_B
        best_idx = torch.argmin(obj_values).item()
        index[j] = best_idx
        idx_tensor[j] = best_idx 
        # print(f"  Pos {j}: switched to {best_idx}, val={obj_values[best_idx]}")
    return index

def algorithm_run(X, c, k, W_init, P_init):
    d, n = X.shape
    w_sum = torch.sum(torch.abs(W_init), dim=1)
    index = torch.nonzero(w_sum, as_tuple=False).squeeze(1).cpu().numpy().tolist()
    if len(index) < k:
        all_idx = set(range(d))
        curr = set(index)
        remain = list(all_idx - curr)
        index.extend(remain[:k-len(index)])
    elif len(index) > k:
        index = index[:k]
    V = W_init[index, :] # (k, c)
    Wt = torch.zeros((d, c), device=device)
    Wt[index, :] = V
    P = P_init.clone()
    obj_history = [total_objective_new_yi(Wt, X, P)]
    print('random_init:',obj_history[0])
    # 预 A = X * X'
    A = torch.mm(X, X.t()) # (d, d)
    max_iter = 100
    start_time = time.time()
    for ii in range(max_iter):
        # Solving V
        B = torch.mm(A, P) # (d, c)
        idx_tensor = torch.tensor(index, device=device, dtype=torch.long)
        STAS = A[idx_tensor][:, idx_tensor] # (k, k)
        tempS = STAS + 1e-7 * torch.eye(k, device=device)
        B_subset = B[idx_tensor, :]
        # V = torch.linalg.solve(tempS, B_subset) # (k, c)
        V =  torch.linalg.lstsq(tempS, B_subset).solution # (k, c)
        VVT = torch.mm(V, V.t())
        # Solving S
        index = solve_feature_selection_vectorized(A, B, V, VVT, index, k)
        Wt.zero_()
        Wt[index, :] = V
        # temp_p = Wt' * X * X' = Wt' * A
        temp_p = torch.mm(Wt.t(), A) # (c, d)
        #  Solving P
        U_p, _, Vh_p = torch.linalg.svd(temp_p, full_matrices=False)
        P = torch.mm(Vh_p.t(), U_p.t()) # (d, c)
        # Check Convergence
        curr_obj = total_objective_new_yi(Wt, X, P)
        # print(f"Iter {ii}, Obj: {curr_obj:.4f}")
        obj_history.append(curr_obj)
        if ii > 4:
            # Check relative error for last 3 steps
            diff1 = abs((obj_history[-1] - obj_history[-2]) / (obj_history[-2] + 1e-12))
            diff2 = abs((obj_history[-2] - obj_history[-3]) / (obj_history[-3] + 1e-12))
            diff3 = abs((obj_history[-3] - obj_history[-4]) / (obj_history[-4] + 1e-12))
            if diff1 <= 1e-4 and diff2 <= 1e-4 and diff3 <= 1e-4:
                print(f"Converged at iter {ii}")
                break
    elapsed = time.time() - start_time
    return index, Wt, obj_history, elapsed

def l20_rc(npy_dir,kept_numer):
    filename = npy_dir
    fea_list = [kept_numer] # feature numbers to test
    
    if os.path.exists(filename):
        X_raw  =  np.load(filename)
        # X_raw = data['X'] # (n, d) 
        X_raw = X_raw.astype(np.float32)
        print(X_raw.shape)
    target_dim = int((2 / 3) * X_raw.shape[1])
    c_val = min(500, target_dim)
    lowd = [c_val]
    maxiter = 1 # 
    X_tensor = torch.from_numpy(X_raw).to(device)
    X_norm = standardization(X_tensor.t()) 
    X = X_norm # (d, n)
    d, n = X.shape
    print(f"Data Loaded. Shape (d, n): {d}, {n}")
    for c in lowd:
        print(f"Testing for dimension c={c}")
        P = torch.eye(d, c, device=device)
        for k in fea_list:
            print(f"  Testing for feature num k={k}")
            min_obj = float('inf')
            best_results = None
            for i in range(maxiter):
                print(f"    Random Init {i+1}")
                W_init = gen_initialization(k, d, c)
                idx, W_final, obj_curve, time_spent = algorithm_run(X, c, k, W_init, P)
                final_obj = obj_curve[-1]
                print(f"    Finished. Time: {time_spent:.2f}s, Final Obj: {final_obj:.4f}")
                if final_obj < min_obj:
                    min_obj = final_obj
                    best_results = {
                        'index': idx,
                        'W': W_final.cpu().numpy(),
                        'obj': final_obj,
                        'curve': obj_curve
                    }
            idx.sort()
            print(idx)
            # print(f"  Best Obj for k={k}: {min_obj:.4f}")
    return idx
if __name__ == '__main__':
    l20_rc("feature_rank_conv_1.npy",50)