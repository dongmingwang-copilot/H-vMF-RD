import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, auc
from scipy.ndimage import gaussian_filter
from skimage import measure
from tqdm import tqdm
from PIL import Image

# ================= 1. 数据集逻辑 =================

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_name, transform, gt_transform, phase='train'):
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        self.base_path = os.path.join(root, class_name)
        
        if phase == 'train':
            self.img_path = os.path.join(self.base_path, 'train', 'good')
            self.img_paths = sorted([os.path.join(self.img_path, f) for f in os.listdir(self.img_path) if f.endswith('.png')])
        else:
            self.img_path = os.path.join(self.base_path, 'test')
            self.gt_path = os.path.join(self.base_path, 'ground_truth')
            self.img_paths, self.gt_paths, self.labels = self._load_test_data()

    def _load_test_data(self):
        img_paths, gt_paths, labels = [], [], []
        defect_types = sorted(os.listdir(self.img_path))
        for dt in defect_types:
            d_path = os.path.join(self.img_path, dt)
            paths = sorted([os.path.join(d_path, f) for f in os.listdir(d_path) if f.endswith('.png')])
            img_paths.extend(paths)
            if dt == 'good':
                gt_paths.extend([None] * len(paths))
                labels.extend([0] * len(paths))
            else:
                g_path = os.path.join(self.gt_path, dt)
                gt_paths.extend([os.path.join(g_path, f.replace('.png', '_mask.png')) for f in os.listdir(d_path)])
                labels.extend([1] * len(paths))
        return img_paths, gt_paths, labels

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, 0
        else:
            gt_path = self.gt_paths[idx]
            # 确保 GT 是严格二值化的张量
            if gt_path is None or not os.path.exists(gt_path):
                gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
            else:
                gt = self.gt_transform(Image.open(gt_path))
                gt = (gt > 0.5).float() # 强制二值化
            return img, gt, self.labels[idx]

def get_perlin_mask(size=256):
    low_res = torch.randn(1, 1, size // 16, size // 16)
    mask = F.interpolate(low_res, size=(size, size), mode='bicubic', align_corners=True)
    mask = (mask > torch.quantile(mask, 0.8)).float()
    return mask

# ================= 2. 数学逻辑：Gaussian Student Head =================

class GaussianStudent(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        # 微架构与 vMF 学生网络保持一致
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.GELU()
        )
        self.mu_head = nn.Conv2d(in_channels, in_channels, 1)
        self.log_var_head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        feat = self.decoder(x)
        mu = self.mu_head(feat)
        # log_var 用于数值稳定性，避免直接预测 sigma^2 产生负值或爆炸
        log_var = torch.clamp(self.log_var_head(feat), -10, 5) 
        return mu, log_var

# ================= 3. 评价指标计算 (修正二值化错误) =================

def compute_pro(masks, amaps, num_th=200):
    masks = masks.cpu().numpy().astype(np.uint8)
    amaps = amaps.cpu().numpy()
    if np.max(masks) == 0: return 0.0
    
    labeled_mask = measure.label(masks, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    thresholds = np.linspace(amaps.min(), amaps.max(), num_th)
    pro_curve, fpr_curve = [], []
    
    for th in thresholds:
        binary_amap = (amaps > th).astype(np.uint8)
        region_probs = []
        for region in regions:
            tp = binary_amap[labeled_mask == region.label].sum()
            region_probs.append(tp / region.area)
        pro_curve.append(np.mean(region_probs))
        
        inverse_mask = 1 - masks
        fpr = (binary_amap * inverse_mask).sum() / inverse_mask.sum()
        fpr_curve.append(fpr)
        
    fpr_curve = np.array(fpr_curve)
    pro_curve = np.array(pro_curve)
    idx = fpr_curve <= 0.3
    if not np.any(idx): return 0.0
    return auc(fpr_curve[idx] / max(fpr_curve[idx]), pro_curve[idx])

# ================= 4. 主实验逻辑 =================

def run_experiment(class_name, root_path, device):
    print(f"\n>>> Gaussian Baseline Training: {class_name}")
    
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gt_trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    train_loader = DataLoader(MVTecDataset(root_path, class_name, trans, gt_trans, 'train'), batch_size=16, shuffle=True)
    test_loader = DataLoader(MVTecDataset(root_path, class_name, trans, gt_trans, 'test'), batch_size=1, shuffle=False)

    # Teacher: WideResNet50-2
    encoder = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    encoder.eval()
    
    def get_layer2_features(x):
        x = encoder.conv1(x); x = encoder.bn1(x); x = encoder.relu(x); x = encoder.maxpool(x)
        x = encoder.layer1(x); return encoder.layer2(x)

    C_FEAT = 512 
    student = GaussianStudent(in_channels=C_FEAT).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(50):
        student.train()
        for img, _ in train_loader:
            img = img.to(device)
            mask = get_perlin_mask(256).to(device)
            corrupted_img = img * (1 - mask) # 结构化破坏
            
            with torch.no_grad():
                target = get_layer2_features(img)
            
            mu, log_var = student(get_layer2_features(corrupted_img))
            
            # 数学逻辑：Isotropic Gaussian NLL Loss
            # L = 0.5 * [ C * log(sigma^2) + ||target - mu||^2 / sigma^2 ]
            dist_sq = torch.sum((target - mu)**2, dim=1, keepdim=True)
            loss = 0.5 * (C_FEAT * log_var + dist_sq / torch.exp(log_var))
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f">>> Evaluating: {class_name}")
    student.eval()
    all_amaps, all_gts, img_scores, img_labels = [], [], [], []
    
    with torch.no_grad():
        for img, gt, label in test_loader:
            img = img.to(device)
            target = get_layer2_features(img)
            mu, log_var = student(target)
            
            # 异常分数的数学逻辑：负对数似然能量
            dist_sq = torch.sum((target - mu)**2, dim=1, keepdim=True)
            score_map = 0.5 * (C_FEAT * log_var + dist_sq / torch.exp(log_var))
            
            score_map = F.interpolate(score_map, size=256, mode='bilinear', align_corners=True)
            score_map = gaussian_filter(score_map.cpu().squeeze().numpy(), sigma=4)
            
            all_amaps.append(torch.from_numpy(score_map))
            all_gts.append((gt.squeeze() > 0.5).int()) # 关键修复：强制二值化
            img_scores.append(np.max(score_map))
            img_labels.append(label)

    # 合并数据进行指标计算
    y_true_pixel = torch.stack(all_gts).flatten().numpy() # 确保是 0/1 整数
    y_score_pixel = torch.stack(all_amaps).flatten().numpy()

    i_auroc = roc_auc_score(img_labels, img_scores)
    p_auroc = roc_auc_score(y_true_pixel, y_score_pixel) # 现在这里不会报错了
    pro = compute_pro(torch.stack(all_gts), torch.stack(all_amaps))
    
    return i_auroc, p_auroc, pro

if __name__ == "__main__":
    DATA_ROOT = "./mvtec_ad" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASSES = ['wood','carpet','grid','transistor','capsule'] # 这里建议先跑一个类验证

    results = []
    for c in CLASSES:
        metrics = run_experiment(c, DATA_ROOT, DEVICE)
        results.append([c] + [round(m * 100, 2) for m in metrics])
    
    df = pd.DataFrame(results, columns=['Class', 'I-AUROC (%)', 'P-AUROC (%)', 'PRO (%)'])
    print("\n" + "="*50)
    print("GAUSSIAN BASELINE (FIXED) RESULTS")
    print("="*50)
    print(df.to_string(index=False))