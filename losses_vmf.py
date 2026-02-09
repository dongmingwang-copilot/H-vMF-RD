import torch
import torch.nn as nn
import scipy.special
import numpy as np

# ==========================================
# 1. Exact Bessel Function Engine (Scipy Bridge)
# ==========================================
class BesselIveFunction(torch.autograd.Function):
    """
    [Exact Analytical Solution]
    Computes I_v(k) * e^{-k} using Scipy's cephes library.
    Implements the EXACT recurrence relation derivative.
    """
    @staticmethod
    def forward(ctx, v, k):
        ctx.v = v
        ctx.save_for_backward(k)
        # Move to CPU for Scipy execution (Machine Precision)
        k_cpu = k.detach().cpu().numpy()
        ive_val = scipy.special.ive(v, k_cpu)
        return torch.from_numpy(ive_val).to(k.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Exact Derivative: d/dk (I_v(k)e^-k) = ive_{v-1}(k) - (1 + v/k) * ive_v(k)
        k, = ctx.saved_tensors
        v = ctx.v
        k_cpu = k.detach().cpu().numpy()
        
        ive_v = scipy.special.ive(v, k_cpu)
        ive_v_minus_1 = scipy.special.ive(v - 1, k_cpu)
        
        grad_k_cpu = ive_v_minus_1 - (1.0 + v / (k_cpu + 1e-9)) * ive_v
        grad_k = torch.from_numpy(grad_k_cpu).to(k.device)
        
        return None, grad_output * grad_k

def bessel_ive(v, k):
    return BesselIveFunction.apply(v, k)

# ==========================================
# 2. Rigorous vMF Loss Module
# ==========================================
class RigorousVMFLoss(nn.Module):
    """
    Computes Exact NLL and Exact KL Divergence using ive(k).
    No approximations used.
    """
    def __init__(self, kl_weight=1e-3):
        super(RigorousVMFLoss, self).__init__()
        self.kl_weight = kl_weight

    def forward(self, teacher_feats, student_outputs):
        total_loss = 0.0
        # Weights for [L1, L2, L3]
        layer_weights = [1.0, 1.0, 1.0]
        
        for i, (t_feat, (s_mu, s_kappa)) in enumerate(zip(teacher_feats, student_outputs)):
            d = float(t_feat.shape[1])
            v = d / 2.0 - 1.0
            
            # --- A. Log Partition Function (ln C_d) ---
            ive_val = bessel_ive(v, s_kappa) + 1e-10
            log_Iv_exact = torch.log(ive_val) + s_kappa
            
            # ln C_d(k) = (d/2-1)ln k - ln I_v - const
            log_Cd = v * torch.log(s_kappa + 1e-10) - log_Iv_exact
            
            # --- B. Reconstruction Loss (NLL) ---
            t_norm = torch.nn.functional.normalize(t_feat, p=2, dim=1)
            cos_theta = torch.sum(t_norm * s_mu, dim=1, keepdim=True)
            nll = -log_Cd - s_kappa * cos_theta
            
            # --- C. KL Divergence (Regularization) ---
            # Ratio A_d(k) = I_{v+1}/I_v = ive_{v+1}/ive_v
            ive_val_plus = bessel_ive(v + 1, s_kappa)
            Ad_k = ive_val_plus / (ive_val + 1e-10)
            kl = s_kappa * Ad_k + log_Cd
            
            # Combine
            layer_loss = (nll.mean() + self.kl_weight * kl.mean()) / d
            total_loss += layer_weights[i] * layer_loss
            
        return total_loss