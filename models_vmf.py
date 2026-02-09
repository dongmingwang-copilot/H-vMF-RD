import torch
import torch.nn as nn
import torch.nn.functional as F
from de_resnet import de_wide_resnet50_2

# ==========================================
# 1. Manifold Consensus Module (MCM)
# [Innovation] A Gated Multi-Scale Fusion Unit
# Replaces generic fusion with a Global-to-Local Consensus mechanism.
# ==========================================
class ManifoldConsensusModule(nn.Module):
    """
    [Architecture Innovation]
    Manifold Consensus Module (MCM).
    Uses Deep (L3) features to filter Shallow (L1) texture features via Gating.
    Prevents the student from relying on local shortcuts.
    """
    def __init__(self, dims=[256, 512, 1024]):
        super(ManifoldConsensusModule, self).__init__()
        
        # Projectors to a common manifold embedding space
        embedding_dim = 128
        self.proj_l1 = nn.Sequential(nn.Conv2d(dims[0], embedding_dim, 1), nn.BatchNorm2d(embedding_dim))
        self.proj_l2 = nn.Sequential(nn.Conv2d(dims[1], embedding_dim, 1), nn.BatchNorm2d(embedding_dim))
        self.proj_l3 = nn.Sequential(nn.Conv2d(dims[2], embedding_dim, 1), nn.BatchNorm2d(embedding_dim))
        
        # Consensus Aggregator (Global Context)
        self.aggregator = nn.Sequential(
            nn.Conv2d(embedding_dim * 3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, dims[0] + dims[1] + dims[2], 1),
            nn.Sigmoid() # Output: [0, 1] Consensus Gates
        )
        
    def forward(self, feats):
        # feats: [f1(256), f2(512), f3(1024)]
        f1, f2, f3 = feats
        
        # 1. Align features to L1 resolution for consensus calculation
        target_h, target_w = f1.shape[2], f1.shape[3]
        
        p1 = self.proj_l1(f1)
        p2 = F.interpolate(self.proj_l2(f2), size=(target_h, target_w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.proj_l3(f3), size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # 2. Generate Global Consensus Gates
        concat_context = torch.cat([p1, p2, p3], dim=1) # [B, 384, H, W]
        gates_all = self.aggregator(concat_context)
        
        # 3. Split gates for each scale
        g1, g2, g3 = torch.split(gates_all, [f1.shape[1], f2.shape[1], f3.shape[1]], dim=1)
        
        # 4. Resize gates back to original resolutions
        g2 = F.interpolate(g2, size=f2.shape[2:], mode='bilinear', align_corners=False)
        g3 = F.interpolate(g3, size=f3.shape[2:], mode='bilinear', align_corners=False)
        
        # 5. Apply Consensus (Residual Gating)
        f1_refined = f1 * (1.0 + g1)
        f2_refined = f2 * (1.0 + g2)
        f3_refined = f3 * (1.0 + g3)
        
        return [f1_refined, f2_refined, f3_refined]

# ==========================================
# 2. Hierarchical vMF Head
# ==========================================
class HierarchicalVMFHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HierarchicalVMFHead, self).__init__()
        
        # Mu Branch
        self.conv_mu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
        )
        
        # Kappa Branch
        self.conv_kappa_feat = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True)
        )
        self.conv_kappa_fusion = nn.Conv2d(in_channels // 2 + 1, 1, 1, bias=True)
        
        # Init for stability
        nn.init.constant_(self.conv_kappa_fusion.bias, 2.0)

    def forward(self, x, context_kappa=None):
        mu = F.normalize(self.conv_mu(x), p=2, dim=1)
        
        feat_k = self.conv_kappa_feat(x)
        if context_kappa is None:
            context_kappa = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        else:
            context_kappa = F.interpolate(context_kappa, size=x.shape[2:], mode='bilinear', align_corners=False)
            context_kappa = context_kappa.detach()
            
        k_in = torch.cat([feat_k, context_kappa], dim=1)
        k_un = self.conv_kappa_fusion(k_in)
        
        # [STAGE 1: HARD KAPPA CLAMP]
        # Prevent Dirac Collapse. 
        # Max kappa reduced from 5000.0 -> 80.0 (Matches your report strategy)
        kappa = F.softplus(k_un) + 1.0
        kappa = torch.clamp(kappa, max=80.0) 
        
        return mu, kappa

# ==========================================
# 3. Student Network (Corrected Data Flow)
# ==========================================
class HVMF_Student(nn.Module):
    def __init__(self):
        super(HVMF_Student, self).__init__()
        
        self.backbone = de_wide_resnet50_2(pretrained=False)
        self.consensus = ManifoldConsensusModule(dims=[256, 512, 1024])
        
        # Heads matching WideResNet channels [256, 512, 1024]
        self.head1 = HierarchicalVMFHead(256, 256)   # L1 (Shallow/HighRes)
        self.head2 = HierarchicalVMFHead(512, 512)   # L2 (Mid)
        self.head3 = HierarchicalVMFHead(1024, 1024) # L3 (Deep/LowRes)

    def forward(self, x):
        # 1. Unpack Input
        if isinstance(x, list):
            bottleneck_feat = x[0]
        else:
            bottleneck_feat = x
            
        # 2. Decode -> [256, 512, 1024] (Based on de_resnet structure)
        # Note: de_resnet usually outputs [HighRes, Mid, LowRes] or vice versa.
        # Based on typical U-Net/Decoder logic, the output list aligns with encoder stages.
        # We assume decoded_feats = [f_high, f_mid, f_low] = [256, 512, 1024]
        decoded_feats = self.backbone(bottleneck_feat)
        
        # Unpack
        f_high, f_mid, f_low = decoded_feats 
        f1, f2, f3 = f_high, f_mid, f_low
        
        # 3. Apply Manifold Consensus
        f1_r, f2_r, f3_r = self.consensus([f1, f2, f3])
        
        # 4. Predict (Top-Down Uncertainty)
        # Deepest (L3, 1024, 16x16) provides context
        mu3, k3 = self.head3(f3_r, context_kappa=None)
        
        # Mid (L2, 512, 32x32)
        mu2, k2 = self.head2(f2_r, context_kappa=k3)
        
        # Shallow (L1, 256, 64x64)
        mu1, k1 = self.head1(f1_r, context_kappa=k2)
        
        # [CRITICAL FIX] 
        # Teacher returns [Layer1(64x64), Layer2(32x32), Layer3(16x16)]
        # We MUST return [(mu1, k1), (mu2, k2), (mu3, k3)] to match.
        return [(mu1, k1), (mu2, k2), (mu3, k3)]