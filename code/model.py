"""Directional denoising and likelihood reconstruction on a Dinomaly decoder."""
from functools import partial
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammaln, ive
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parent / "third_party" / "dinomaly"
sys.path.insert(0, str(UPSTREAM))
from models.vision_transformer import Block, bMlp, LinearAttention2


class StableLinearAttention(LinearAttention2):
    """Accumulate the positive linear-attention kernel in float32."""

    def forward(self, x, attn_mask=None):
        batch, tokens, dimension = x.shape
        qkv = self.qkv(x).reshape(batch,tokens,3,self.num_heads,dimension//self.num_heads).permute(2,0,3,1,4)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q,k,v = qkv.float().unbind(0)
            # This equals ELU(x)+1 without cancellation for negative inputs.
            q = torch.exp(q.clamp(max=0)) + q.clamp(min=0)
            k = torch.exp(k.clamp(max=0)) + k.clamp(min=0)
            kv = torch.einsum('...sd,...se->...de',k,v)
            denominator = torch.einsum('...sd,...d->...s',q,k.sum(dim=-2)).clamp_min(1e-8)
            result = torch.einsum('...de,...sd,...s->...se',kv,q,denominator.reciprocal())
        result = result.transpose(1,2).reshape(batch,tokens,dimension).to(qkv.dtype)
        return self.proj_drop(self.proj(result)),kv


class LogPartition(torch.autograd.Function):
    """log(C_d(0)/C_d(kappa)) with a float64 scaled-Bessel evaluation."""

    @staticmethod
    def forward(ctx, kappa, dimension):
        values = kappa.detach().double().cpu().numpy()
        nu = dimension / 2 - 1
        scaled = ive(nu, values)
        ratio = ive(nu + 1, values) / scaled
        if not np.all(np.isfinite(ratio)) or not np.all(scaled > 0):
            raise FloatingPointError("Invalid Bessel evaluation in declared concentration range")
        log_z = np.log(scaled) + values - nu * np.log(values) + nu * np.log(2) + gammaln(nu + 1)
        ctx.save_for_backward(torch.as_tensor(ratio, device=kappa.device, dtype=kappa.dtype))
        return torch.as_tensor(log_z, device=kappa.device, dtype=kappa.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (ratio,) = ctx.saved_tensors
        return gradient * ratio, None


def log_partition(kappa, dimension):
    return LogPartition.apply(kappa, dimension)


def directional_moments(kappa, dimension):
    values = kappa.detach().double().cpu().numpy()
    nu = dimension/2-1
    mean = ive(nu+1,values)/ive(nu,values)
    variance = 1-mean**2-(dimension-1)*mean/values
    if not np.all(variance>0):
        raise FloatingPointError('Invalid directional variance')
    return (torch.as_tensor(mean,device=kappa.device,dtype=torch.float32),
            torch.as_tensor(variance,device=kappa.device,dtype=torch.float32))


def mixture_nll(log_densities, log_weights, dimension, coupled=False):
    """Combine B x H x P x K log densities with image-level component priors."""
    if coupled:
        joint = log_densities.sum(dim=1) + log_weights[:, None]
        return -torch.logsumexp(joint, dim=-1).unsqueeze(1) / (dimension * log_densities.shape[1])
    independent = log_densities + log_weights[:, None, None]
    return -torch.logsumexp(independent, dim=-1) / dimension


def directional_patch_corruption(teacher, generator=None, probability=1.0):
    """Mix two local regions geodesically with another normal image."""
    batch, groups, tokens, dimension = teacher.shape
    side = math.isqrt(tokens - 5)
    assert side * side == tokens - 5
    if batch < 2:
        return teacher
    spatial = teacher[:, :, 5:]
    norm = spatial.norm(dim=-1, keepdim=True)
    direction = F.normalize(spatial, dim=-1)
    donor = direction.roll(1, dims=0)
    midpoint = F.normalize(direction + donor, dim=-1)
    midpoint = torch.where((direction + donor).norm(dim=-1, keepdim=True) > 1e-6, midpoint, direction)
    width = max(1, side // 4)
    origins = torch.randint(side - width + 1, (batch, 2, 2), generator=generator, device=teacher.device)
    y, x = torch.meshgrid(torch.arange(side, device=teacher.device), torch.arange(side, device=teacher.device), indexing='ij')
    mask = ((y[None,None] >= origins[:,:,0,None,None]) & (y[None,None] < origins[:,:,0,None,None] + width)
            & (x[None,None] >= origins[:,:,1,None,None]) & (x[None,None] < origins[:,:,1,None,None] + width)).any(1)
    if not 0<=probability<=1:
        raise ValueError('Corruption probability must lie in [0,1]')
    if probability<1:
        selected = torch.rand(batch,generator=generator,device=teacher.device)<probability
        mask = mask & selected[:,None,None]
    corrupted = torch.where(mask.reshape(batch,1,side*side,1), midpoint * norm, spatial)
    return torch.cat([teacher[:,:,:5], corrupted], dim=2)


class DirectionalHead(nn.Module):
    def __init__(self, dimension, components):
        super().__init__()
        self.dimension = dimension
        self.components = components
        self.offset = nn.Linear(dimension, components * dimension)
        self.concentration = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, components))

    def forward(self, features):
        batch, tokens, dimension = features.shape
        offsets = self.offset(features).reshape(batch, tokens, self.components, dimension)
        mean = F.normalize(features.unsqueeze(2) + offsets, dim=-1)
        context = features[:, 5:].mean(dim=1)
        logits = self.concentration(context).float()
        kappa = dimension * (0.25 + 15.75 * torch.sigmoid(logits))
        return mean[:, 5:], kappa


class DirectionalRD(nn.Module):
    VARIANTS = ("dinomaly", "vmf_fixed", "vmf_single", "vmf_mixture", "context_mixture", "coupled_mixture",
                "denoising_cosine", "denoising_vmf", "denoising_cosine_p25", "denoising_vmf_p25")

    def __init__(self, variant="denoising_vmf_p25", dimension=384):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.directional = variant not in {'dinomaly', 'denoising_cosine', 'denoising_cosine_p25'}
        self.corruption_probability = 0.25 if variant.endswith('_p25') else 1.0
        self.dimension = dimension
        self.components = 1 if variant in {'vmf_fixed', 'vmf_single', 'denoising_vmf', 'denoising_vmf_p25'} else 3
        self.bottleneck = bMlp(dimension, dimension * 4, dimension, drop=0.2)
        self.decoder = nn.ModuleList([
            Block(dim=dimension, num_heads=dimension // 64, mlp_ratio=4., qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=StableLinearAttention)
            for _ in range(8)
        ])
        if self.directional:
            self.heads = nn.ModuleList([DirectionalHead(dimension, self.components) for _ in range(2)])
            if variant in {"context_mixture", "coupled_mixture"}:
                self.router = nn.Sequential(nn.LayerNorm(dimension * 2), nn.Linear(dimension * 2, dimension // 2),
                                            nn.GELU(), nn.Linear(dimension // 2, self.components))
        self.apply(self.initialize)
        if self.directional:
            for head in self.heads:
                nn.init.normal_(head.offset.weight, std=0.005)
                nn.init.normal_(head.offset.bias, std=0.01)
                nn.init.zeros_(head.concentration[-1].weight)
                nn.init.constant_(head.concentration[-1].bias, math.log(0.75 / 15.0))

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, teacher):
        # Averaging two four-layer groups equals the original eight-layer bottleneck.
        if self.training and self.variant.startswith('denoising_'):
            teacher = directional_patch_corruption(teacher,probability=self.corruption_probability)
        x = self.bottleneck(teacher.mean(dim=1))
        decoded = []
        for block in self.decoder:
            x = block(x)
            decoded.append(x)
        reconstruction = [torch.stack(decoded[4:]).mean(0), torch.stack(decoded[:4]).mean(0)]
        if not self.directional:
            return {"reconstruction": reconstruction}
        heads = [head(features) for head, features in zip(self.heads, reconstruction)]
        if self.variant == "vmf_fixed":
            heads = [(mean, torch.full_like(kappa, self.dimension)) for mean, kappa in heads]
        if self.variant in {"context_mixture", "coupled_mixture"}:
            context = torch.cat([features[:, 5:].mean(1) for features in reconstruction], dim=-1)
            log_weights = F.log_softmax(self.router(context).float(), dim=-1)
        else:
            log_weights = teacher.new_full((teacher.shape[0], self.components), -math.log(self.components)).float()
        return {"reconstruction": reconstruction, "heads": heads, "log_weights": log_weights}

    def nll_maps(self, teacher, output):
        log_weights = output["log_weights"]
        densities = []
        for scale, (mean, kappa) in enumerate(output["heads"]):
            target = F.normalize(teacher[:, scale, 5:].float(), dim=-1)
            cosine = (target.unsqueeze(2) * mean.float()).sum(-1)
            log_z = log_partition(kappa, self.dimension)
            densities.append(kappa[:, None] * cosine - log_z[:, None])
        return mixture_nll(torch.stack(densities, dim=1), log_weights, self.dimension,
                           coupled=self.variant == "coupled_mixture")

    def loss(self, teacher, output, step):
        if not self.directional:
            return hard_mining_cosine(teacher, output["reconstruction"], step)
        likelihood = self.nll_maps(teacher, output).mean()
        # A fixed short warm-up aligns means before fitting mixture concentration.
        warmup = max(0., 1. - step / 500.)
        directional = torch.stack([
            (1 - F.cosine_similarity(teacher[:, scale, 5:].float(), features[:, 5:].float(), dim=-1)).mean()
            for scale, features in enumerate(output["reconstruction"])
        ]).mean()
        weights = output["log_weights"].exp().mean(0)
        balance = (weights * (weights.clamp_min(1e-12).log() + math.log(self.components))).sum()
        return (1. - warmup) * likelihood + warmup * directional + 0.001 * balance

    def anomaly_map(self, teacher, output, score="default"):
        if score in {"angular_standardized","angular_deviation"}:
            assert self.variant in {'vmf_single','denoising_vmf','denoising_vmf_p25'}
            maps = []
            for scale,(mean,kappa) in enumerate(output['heads']):
                target = F.normalize(teacher[:,scale,5:].float(),dim=-1)
                cosine = (target*mean[:,:,0].float()).sum(-1)
                expected,variance = directional_moments(kappa[:,0],self.dimension)
                if score == "angular_standardized":
                    maps.append((expected[:,None]-cosine)/variance.sqrt()[:,None])
                else:
                    maps.append(kappa[:,0,None]*(expected[:,None]-cosine)/self.dimension)
            scores = torch.stack(maps).mean(0)
        elif not self.directional or score == "cosine":
            maps = [1 - F.cosine_similarity(teacher[:, scale, 5:].float(), features[:, 5:].float(), dim=-1)
                    for scale, features in enumerate(output["reconstruction"])]
            scores = torch.stack(maps, dim=1).mean(1)
        else:
            scores = self.nll_maps(teacher, output).mean(1)
        side = math.isqrt(scores.shape[-1])
        assert side * side == scores.shape[-1]
        return scores.reshape(-1, side, side)


def hard_mining_cosine(teacher, decoded, step):
    losses = []
    proportion = min(0.9 * step / 1000, 0.9)
    for scale, features in enumerate(decoded):
        target = teacher[:, scale, 5:].detach().float()
        student = features[:, 5:].float()
        if torch.is_grad_enabled() and student.requires_grad:
            with torch.no_grad():
                residual = 1 - F.cosine_similarity(target, student, dim=-1)
                count = max(1, int(residual.numel() * (1 - proportion)))
                threshold = torch.topk(residual.flatten(), count).values[-1]
                multiplier = torch.where(residual < threshold, 0.1, 1.).unsqueeze(-1)
            student.register_hook(lambda gradient, multiplier=multiplier: gradient * multiplier)
        losses.append((1 - F.cosine_similarity(target.flatten(1), student.flatten(1), dim=1)).mean())
    return torch.stack(losses).mean()
