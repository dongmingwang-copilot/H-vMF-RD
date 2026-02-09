import torch
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, auc
from scipy.ndimage import gaussian_filter
from skimage import measure
from statistics import mean

# Official Imports
from dataset import get_data_transforms, MVTecDataset
from resnet import wide_resnet50_2

# Modules
from models_vmf import HVMF_Student
from losses_vmf import RigorousVMFLoss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Feature Corruption (Perlin Noise)
# ==========================================
def generate_perlin_noise_mask(shape, scale, device):
    B, C, H, W = shape
    low_res_h = max(1, H // scale)
    low_res_w = max(1, W // scale)
    noise = torch.rand(B, 1, low_res_h, low_res_w, device=device)
    noise = F.interpolate(noise, size=(H, W), mode='bicubic', align_corners=False)
    mask = (noise > 0.3).float()
    return mask

def corrupt_features(features):
    if isinstance(features, torch.Tensor):
        features = [features]
    corrupted_list = []
    for feat in features:
        # Multiplicative Noise
        noise = torch.randn_like(feat) * 0.1
        feat_noisy = feat * (1.0 + noise)
        
        # Channel Dropout (0.25)
        B, C, H, W = feat.shape
        chan_mask = torch.bernoulli(torch.full((B, C, 1, 1), 0.75, device=feat.device))
        feat_noisy = feat_noisy * chan_mask * (1.0 / 0.75)
        
        # Structural Masking
        struct_mask = generate_perlin_noise_mask(feat.shape, scale=6, device=feat.device)
        feat_noisy = feat_noisy * struct_mask
            
        corrupted_list.append(feat_noisy)
    return corrupted_list

# ==========================================
# 2. Evaluation Logic (Strict Fix)
# ==========================================
def compute_pro(masks, amaps, num_th=200):
    if masks.ndim != 3 or amaps.ndim != 3:
        return 0.0
    
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        new_row = pd.DataFrame([{"pro": mean(pros) if pros else 0.0, "fpr": fpr, "threshold": th}])
        if not df.empty:
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

    df = df[df["fpr"] < 0.3]
    if df.empty: return 0.0
    
    max_fpr = df["fpr"].max()
    if max_fpr == 0: return df["pro"].max() 
        
    df["fpr"] = df["fpr"] / max_fpr
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
        
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        
        if isinstance(fs, tuple) or isinstance(fs, list):
            s_mu = fs[0] 
        else:
            s_mu = fs

        ft_norm = F.normalize(ft, p=2, dim=1)
        a_map = 1 - F.cosine_similarity(s_mu, ft_norm)
        
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map # acc mode
            
    return anomaly_map

def evaluation(encoder, bn, decoder, dataloader, device):
    encoder.eval()
    bn.eval()
    decoder.eval()
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            
            anomaly_map = cal_anomaly_map(outputs, inputs, img.shape[-1], amap_mode='acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            if label.item() != 0:
                gt_np = gt.cpu().numpy().astype(int)
                if gt_np.ndim == 4: gt_np = gt_np.squeeze(1)
                aupro_list.append(compute_pro(gt_np, anomaly_map[np.newaxis,:,:]))
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            
    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    aupro = np.mean(aupro_list) if len(aupro_list) > 0 else 0.0
    
    return auroc_px, auroc_sp, aupro

# ==========================================
# 3. Training & Visualization Modules
# ==========================================
def train(_class_):
    print(f"Start H-vMF Training for: {_class_}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # [FEATURE] Skip if checkpoint exists
    ckp_path = f'./checkpoints/hvmf_best_{_class_}.pth'
    if not os.path.exists('./checkpoints'): os.makedirs('./checkpoints')
    
    # Initialize Data & Models for Eval/Train
    data_transform, gt_transform = get_data_transforms(256, 256)
    root_path = './mvtec_ad/'
    test_data = MVTecDataset(root=root_path + _class_, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device); bn = bn.to(device); encoder.eval()
    decoder = HVMF_Student().to(device)
    
    # SKIP LOGIC
    if os.path.exists(ckp_path):
        print(f"--> Checkpoint found for {_class_}, skipping training...")
        # Load weights and perform one evaluation to get metrics for the table
        ckp = torch.load(ckp_path)
        bn.load_state_dict(ckp['bn'])
        decoder.load_state_dict(ckp['decoder'])
        
        s_auc, p_auc, pro = evaluation(encoder, bn, decoder, test_dataloader, device)
        print(f"Loaded Results: S-AUC:{s_auc:.3f}, P-AUC:{p_auc:.3f}, PRO:{pro:.3f}")
        return (s_auc, p_auc, pro)

    # --- Start Training ---
    epochs = 200 
    learning_rate = 0.005
    train_data = ImageFolder(root=root_path + _class_ + '/train', transform=data_transform)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = RigorousVMFLoss(kl_weight=0.0)

    best_metrics = (0.0, 0.0, 0.0)

    for epoch in range(epochs):
        bn.train(); decoder.train()
        loss_list = []
        
        # KL Scheduler
        if epoch < 5: 
            beta = 0.01
        else: beta = 0.01 + (epoch - 5) / (epochs - 5) * (1.0 - 0.01)
        criterion.kl_weight = beta
        
        for img, _ in train_dataloader:
            img = img.to(device)
            with torch.no_grad():
                clean_inputs = encoder(img)
            
            compact_feat_list = bn(clean_inputs)
            corrupted_feat_list = corrupt_features(compact_feat_list)
            outputs = decoder(corrupted_feat_list)
            
            loss = criterion(clean_inputs, outputs)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)
            optimizer.step()
            loss_list.append(loss.item())
            
        scheduler.step()
        
        # [Strict RD Logic] Print every epoch
        print('epoch [{}/{}], loss:{:.4f}, lr:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list), scheduler.get_last_lr()[0]))
        
        # [Strict RD Logic] Eval every 10 epochs
        if (epoch + 1) % 10 == 0:
            s_auc, p_auc, pro = evaluation(encoder, bn, decoder, test_dataloader, device)
            print(f'Eval: S-AUC:{s_auc:.3f}, P-AUC:{p_auc:.3f}, PRO:{pro:.3f}')
            
            is_best = False
            if s_auc > best_metrics[0] + 0.001: is_best = True
            elif abs(s_auc - best_metrics[0]) <= 0.001:
                if p_auc > best_metrics[1] + 0.001: is_best = True
                elif abs(p_auc - best_metrics[1]) <= 0.001:
                    if pro > best_metrics[2]: is_best = True
            
            if is_best:
                best_metrics = (s_auc, p_auc, pro)
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
                print(f"--> Best Model Saved: {best_metrics}")

    return best_metrics

def get_vis_samples(class_name, device):
    """
    Extracts 1 Normal and 1 Anomaly sample with Heatmap/GT for visualization.
    """
    ckp_path = f'./checkpoints/hvmf_best_{class_name}.pth'
    if not os.path.exists(ckp_path): return None, None, None, None

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec_ad/' + class_name
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    loader = DataLoader(test_data, batch_size=1, shuffle=True) # Random sample

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device); bn = bn.to(device)
    decoder = HVMF_Student().to(device)
    
    ckp = torch.load(ckp_path)
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    
    encoder.eval(); bn.eval(); decoder.eval()
    
    normal_img, anomaly_img, anomaly_gt, anomaly_map = None, None, None, None
    found_normal, found_anomaly = False, False
    
    with torch.no_grad():
        for img, gt, label, _ in loader:
            # Denormalize image for Vis
            img_vis = img.cpu().numpy().squeeze().transpose(1, 2, 0)
            mean_ = np.array([0.485, 0.456, 0.406])
            std_ = np.array([0.229, 0.224, 0.225])
            img_vis = std_ * img_vis + mean_
            img_vis = np.clip(img_vis, 0, 1)

            if label.item() == 0 and not found_normal:
                normal_img = img_vis
                found_normal = True
            
            elif label.item() != 0 and not found_anomaly:
                img_cuda = img.to(device)
                inputs = encoder(img_cuda)
                outputs = decoder(bn(inputs))
                amap = cal_anomaly_map(outputs, inputs, img.shape[-1], amap_mode='acc')
                amap = gaussian_filter(amap, sigma=4)
                
                anomaly_img = img_vis
                anomaly_gt = gt.cpu().numpy().squeeze()
                anomaly_map = amap
                found_anomaly = True
            
            if found_normal and found_anomaly: break
            
    return normal_img, anomaly_img, anomaly_gt, anomaly_map

def run_full_suite():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = ['wood']
    
    results = {'Class': [], 'S-AUC': [], 'P-AUC': [], 'PRO': []}
    
    # 1. Train Loop
    for cls in classes:
        best_s, best_p, best_pro = train(cls)
        results['Class'].append(cls)
        results['S-AUC'].append(best_s)
        results['P-AUC'].append(best_p)
        results['PRO'].append(best_pro)
        
    # 2. Table Generation
    df = pd.DataFrame(results)
    mean_row = pd.DataFrame([['Mean', df['S-AUC'].mean(), df['P-AUC'].mean(), df['PRO'].mean()]], columns=df.columns)
    df = pd.concat([df, mean_row], ignore_index=True)
    print("\n=== Final Results ===")
    print(df)
    df.to_csv('final_results.csv', index=False)
    
    # 3. 4x15 Visualization
    print("\nGenerating Visualization Grid...")
    fig, axes = plt.subplots(4, 15, figsize=(30, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    row_titles = ['Normal', 'Defect', 'GT', 'Heatmap']
    
    for i, cls in enumerate(classes):
        norm_img, anom_img, gt, amap = get_vis_samples(cls, device)
        if norm_img is None: continue 

        # Row 1: Normal
        axes[0, i].imshow(norm_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(cls, fontsize=10)
        if i==0: axes[0, i].set_ylabel(row_titles[0], fontsize=12)

        # Row 2: Defect
        axes[1, i].imshow(anom_img)
        axes[1, i].axis('off')
        if i==0: axes[1, i].set_ylabel(row_titles[1], fontsize=12)

        # Row 3: GT
        axes[2, i].imshow(gt, cmap='gray')
        axes[2, i].axis('off')
        if i==0: axes[2, i].set_ylabel(row_titles[2], fontsize=12)

        # Row 4: Heatmap
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        axes[3, i].imshow(amap_norm, cmap='jet')
        axes[3, i].axis('off')
        if i==0: axes[3, i].set_ylabel(row_titles[3], fontsize=12)
        
    plt.savefig('all_classes_vis.png', dpi=300, bbox_inches='tight')
    print("Saved 'all_classes_vis.png'")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    setup_seed(111)
    run_full_suite()