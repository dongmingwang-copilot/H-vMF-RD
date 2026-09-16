import math

import mpmath
import numpy as np
import pytest
import torch

from model import DirectionalRD, log_partition, mixture_nll, directional_moments, directional_patch_corruption, StableLinearAttention, LinearAttention2


@pytest.mark.parametrize("dimension", [64, 384, 768])
def test_log_partition_matches_high_precision(dimension):
    mpmath.mp.dps = 80
    values = np.array([0.25, 1., 4., 16.]) * dimension
    actual = log_partition(torch.tensor(values, dtype=torch.float64), dimension).numpy()
    nu = dimension / 2 - 1
    expected = np.array([float(mpmath.log(mpmath.besseli(nu, value)) - nu * mpmath.log(value)
                               + nu * mpmath.log(2) + mpmath.loggamma(nu + 1)) for value in values])
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-12)


def test_partition_gradient():
    values = torch.tensor([96., 384., 1536., 6144.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: log_partition(values, 384), (values,), eps=1e-3, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("variant", DirectionalRD.VARIANTS)
def test_training_step_is_finite(variant):
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DirectionalRD(variant).to(device)
    teacher = torch.randn(2, 2, 21, 384, device=device)
    output = model(teacher)
    loss = model.loss(teacher, output, 600)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.anomaly_map(teacher, output).shape == (2, 4, 4)
    model.eval()
    with torch.inference_mode():
        output = model(teacher)
        assert torch.isfinite(model.loss(teacher, output, 600))


def test_shared_context_weights_and_unit_means():
    model = DirectionalRD("context_mixture")
    output = model(torch.randn(2, 2, 21, 384))
    torch.testing.assert_close(output["log_weights"].exp().sum(-1), torch.ones(2))
    assert output["log_weights"].shape == (2, 3)
    for mean, concentration in output["heads"]:
        torch.testing.assert_close(mean.norm(dim=-1), torch.ones(2, 16, 3))
        assert (concentration >= 96).all() and (concentration <= 6144).all()


def test_attention_matches_original_in_regular_range():
    original=LinearAttention2(64,num_heads=1,qkv_bias=True)
    stable=StableLinearAttention(64,num_heads=1,qkv_bias=True)
    stable.load_state_dict(original.state_dict())
    x=torch.randn(2,25,64)
    torch.testing.assert_close(original(x)[0],stable(x)[0],atol=2e-6,rtol=2e-5)


def test_attention_is_finite_for_saturated_negative_queries():
    original=LinearAttention2(64,num_heads=1,qkv_bias=True)
    stable=StableLinearAttention(64,num_heads=1,qkv_bias=True)
    with torch.no_grad():
        original.qkv.weight.zero_()
        original.qkv.bias[:128].fill_(-25.)
    stable.load_state_dict(original.state_dict())
    x=torch.randn(2,25,64,requires_grad=True)
    assert not torch.isfinite(original(x)[0]).all()
    output=stable(x)[0]
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()


def test_joint_density_uses_prior_once_and_matches_direct_product():
    density = torch.tensor([[[[.7,.2],[.3,.8]],[[.9,.1],[.4,.6]]]], dtype=torch.float64)
    weights = torch.tensor([[.25,.75]], dtype=torch.float64)
    actual = mixture_nll(density.log(), weights.log(), dimension=4, coupled=True)[:,0]
    expected = -(density.prod(dim=1)*weights[:,None]).sum(-1).log()/8
    torch.testing.assert_close(actual, expected)


def test_single_component_joint_equals_independent_average():
    density = torch.randn(3,2,7,1, dtype=torch.float64)
    weights = torch.zeros(3,1, dtype=torch.float64)
    joint = mixture_nll(density,weights,384,True).mean(1)
    independent = mixture_nll(density,weights,384,False).mean(1)
    torch.testing.assert_close(joint,independent)


def test_joint_score_detects_inconsistent_component_pairing():
    aligned = torch.tensor([[[[.99,.01]],[[.99,.01]]]], dtype=torch.float64).log()
    inconsistent = aligned.clone()
    inconsistent[:,1] = inconsistent[:,1].flip(-1)
    weights = torch.full((1,2), -math.log(2), dtype=torch.float64)
    torch.testing.assert_close(mixture_nll(aligned,weights,4,False), mixture_nll(inconsistent,weights,4,False))
    assert mixture_nll(inconsistent,weights,4,True).item() > mixture_nll(aligned,weights,4,True).item()


def test_joint_score_is_invariant_to_shared_component_permutation():
    density = torch.randn(3,2,7,3, dtype=torch.float64)
    weights = torch.randn(3,3, dtype=torch.float64).log_softmax(-1)
    permutation = torch.tensor([2,0,1])
    torch.testing.assert_close(mixture_nll(density,weights,384,True),
                              mixture_nll(density[:,:,:,permutation],weights[:,permutation],384,True))


def test_directional_variance_matches_partition_second_derivative():
    values = torch.tensor([96.,384.,1536.,6144.],dtype=torch.float64)
    mean,variance = directional_moments(values,384)
    step = .02
    upper = log_partition(values+step,384)
    lower = log_partition(values-step,384)
    center = log_partition(values,384)
    torch.testing.assert_close(mean.double(),(upper-lower)/(2*step),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(variance.double(),(upper-2*center+lower)/step**2,atol=1e-8,rtol=2e-3)


def test_local_corruption_preserves_norms_prefix_and_input():
    teacher = torch.randn(4,2,405,64)
    original = teacher.clone()
    changed = directional_patch_corruption(teacher,generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(teacher,original)
    torch.testing.assert_close(changed[:,:,:5],teacher[:,:,:5])
    torch.testing.assert_close(changed.norm(dim=-1),teacher.norm(dim=-1))
    mask = (changed[:,:,5:]-teacher[:,:,5:]).abs().sum(-1)>1e-5
    assert torch.equal(mask[:,0],mask[:,1])
    assert (mask[:,0].sum(-1)>=25).all() and (mask[:,0].sum(-1)<=50).all()


def test_local_corruption_has_finite_antipodal_limit():
    teacher = torch.randn(1,2,21,64)
    teacher = torch.cat([teacher,-teacher])
    changed = directional_patch_corruption(teacher)
    assert torch.isfinite(changed).all()
    torch.testing.assert_close(changed.norm(dim=-1),teacher.norm(dim=-1))


def test_zero_corruption_probability_is_exact_identity():
    teacher = torch.randn(8,2,405,64)
    changed = directional_patch_corruption(teacher,probability=0.)
    assert torch.equal(changed,teacher)


def test_reduced_corruption_selects_whole_images_reproducibly():
    teacher = torch.randn(64,2,405,64)
    first = directional_patch_corruption(teacher,torch.Generator().manual_seed(17),probability=.25)
    second = directional_patch_corruption(teacher,torch.Generator().manual_seed(17),probability=.25)
    assert torch.equal(first,second)
    mask = (first[:,:,5:]-teacher[:,:,5:]).abs().sum(-1)>1e-5
    assert torch.equal(mask[:,0],mask[:,1])
    counts = mask[:,0].sum(-1)
    assert ((counts==0)|((counts>=25)&(counts<=50))).all()
    assert 0<int((counts>0).sum())<len(teacher)
    torch.testing.assert_close(first.norm(dim=-1),teacher.norm(dim=-1))

def test_fixed_concentration_preserves_head_capacity_and_cosine_ranking():
    torch.manual_seed(17)
    fixed = DirectionalRD("vmf_fixed", dimension=64).eval()
    learned = DirectionalRD("vmf_single", dimension=64).eval()
    learned.load_state_dict(fixed.state_dict(), strict=True)
    teacher = torch.randn(2, 2, 21, 64)
    with torch.no_grad():
        for head in fixed.heads:
            head.concentration[-1].bias.fill_(3.0)
        output = fixed(teacher)
        matched = learned(teacher)
        discrepancies = []
        for hierarchy, (mean, kappa) in enumerate(output["heads"]):
            assert torch.equal(kappa, torch.full_like(kappa, 64))
            torch.testing.assert_close(mean, matched["heads"][hierarchy][0])
            target = torch.nn.functional.normalize(teacher[:, hierarchy, 5:], dim=-1)
            discrepancies.append(1 - (target * mean[:, :, 0]).sum(-1))
        cosine = torch.stack(discrepancies, dim=1).mean(1)
        constant = (log_partition(torch.tensor(64.), 64) - 64) / 64
        likelihood = fixed.anomaly_map(teacher, output).flatten(1)
        torch.testing.assert_close(likelihood, cosine + constant, atol=2e-6, rtol=2e-6)
    fixed.train()
    loss = fixed.loss(teacher, fixed(teacher), 600)
    loss.backward()
    assert all(head.concentration[-1].weight.grad is None for head in fixed.heads)
    assert all(head.offset.weight.grad is not None for head in fixed.heads)
