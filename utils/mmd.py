import torch

min_var_est = 1e-8

def default_gamma(X:torch.Tensor):
    gamma = 1.0 / X.shape[1]
    return gamma

def rbf_kernel(X:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X)
    K.fill_diagonal_(0) # avoid floating point error
    K.pow_(2)
    K.mul_(-gamma)
    K.exp_()
    return K

def select_prototypes(K:torch.Tensor, num_prototypes:int):
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected = torch.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]

    for i in range(num_prototypes):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]

        if selected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[selected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= (selected.shape[0] + 1)
            s1 -= s2

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order

def select_prototypes_wo_plbl(K:torch.Tensor, num_prototypes:int, plbl_idxs:torch.Tensor, plbl_cratio: float):
    "check wheather it is pseudo label and reject it from comsuming budgets"
    nplbl_ceil = int(plbl_cratio * num_prototypes)
    remained_nprototypes = num_prototypes
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected = torch.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]
    is_plblselected = torch.zeros_like(sample_indices)
    plblselected = sample_indices[is_plblselected > 0]
    is_anyselected = is_selected + is_plblselected # combined indicater for selection
    anyselected = sample_indices[is_anyselected > 0]

    idx = 0
    while 0 < remained_nprototypes:
        candidate_indices = sample_indices[is_anyselected == 0]
        s1 = colsum[candidate_indices]

        if anyselected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[anyselected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= (anyselected.shape[0] + 1)
            s1 -= s2

        best_sample_index = candidate_indices[s1.argmax()]
        
        ### check pseudo label and splitly save each plbls
        if best_sample_index in plbl_idxs and plblselected.shape[0] < nplbl_ceil:
            is_plblselected[best_sample_index] = idx + 1
            plblselected = sample_indices[is_plblselected > 0]
        elif best_sample_index not in plbl_idxs:
            is_selected[best_sample_index] = idx + 1
            selected = sample_indices[is_selected > 0]
            remained_nprototypes -= 1
        else: ### keep num plbl under nplbl_ceil
            is_plblselected[best_sample_index] = idx + 1
            pmin = torch.min(is_plblselected[is_plblselected > 0])
            pmdx = (is_plblselected == pmin).nonzero().item()
            is_plblselected[pmdx] = 0
            plblselected = sample_indices[is_plblselected > 0]
            
            ### move first selected plbl into selected 
            is_selected[pmdx] = pmin
            selected = sample_indices[is_selected > 0]
            remained_nprototypes -= 1

        idx += 1
        is_anyselected = is_selected + is_plblselected # combined indicater for selection
        anyselected = sample_indices[is_anyselected > 0]

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    plblselected_in_order = plblselected[is_plblselected[is_plblselected > 0].argsort()]

    return selected_in_order, plblselected_in_order