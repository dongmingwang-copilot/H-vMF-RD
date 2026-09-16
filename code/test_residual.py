import math
import numpy as np
from sklearn.covariance import oas
import torch

from residual_rd import residual_vectors,residual_score


def test_spherical_log_is_tangent_and_has_geodesic_length():
    angle = .4
    teacher = torch.zeros(1,2,9,4)
    teacher[:,:,5:,0] = math.cos(angle)
    teacher[:,:,5:,1] = math.sin(angle)
    student = torch.zeros(1,9,4)
    student[:,:,0] = 1
    vectors = residual_vectors(teacher,{'reconstruction':[student,student]},True)
    for vector in vectors:
        torch.testing.assert_close(vector[...,0],torch.zeros(1,4))
        torch.testing.assert_close(vector.norm(dim=-1),torch.full((1,4),angle))


def test_zero_directional_error_has_finite_zero_log_map():
    teacher = torch.randn(1,2,9,4)
    output = {'reconstruction':[teacher[:,0],teacher[:,1]]}
    for vector in residual_vectors(teacher,output,True):
        assert torch.isfinite(vector).all()
        torch.testing.assert_close(vector,torch.zeros_like(vector),atol=1e-6,rtol=0)


def test_whitened_score_matches_inverse_covariance_quadratic_form():
    rng = np.random.default_rng(18)
    samples = rng.normal(size=(50,4))*np.array([1.,2.,.2,.5])
    covariance,_ = oas(samples)
    eigenvalues,eigenvectors = np.linalg.eigh(covariance)
    mean = torch.tensor(samples.mean(0),dtype=torch.float32)
    whitening = torch.tensor(eigenvectors/np.sqrt(eigenvalues)[None],dtype=torch.float32)
    teacher = torch.randn(2,2,9,4)
    output = {'reconstruction':[torch.randn(2,9,4),torch.randn(2,9,4)]}
    vectors = residual_vectors(teacher,output,True)
    precision = torch.tensor(np.linalg.inv(covariance),dtype=torch.float32)
    direct = torch.stack([torch.einsum('bpd,de,bpe->bp',v-mean,precision,v-mean)/4 for v in vectors]).mean(0)
    actual = residual_score(teacher,output,[{'mean':mean,'whitening':whitening}]*2,'tangent_full')
    torch.testing.assert_close(actual.reshape(2,4),direct)
