# Structured Directional Denoising Reverse Distillation

SDD-RD reconstructs normalized DINOv2 features with a von Mises-Fisher distribution at each of two reverse-decoder hierarchies. During training, selected local regions receive norm-preserving directional perturbations from another normal image. The reconstruction target remains clean. Perturbation is disabled at inference.

The code includes the deterministic Dinomaly comparison, directional and denoising ablations, and adapters for RD, RD++, and PatchCore. All methods use a common normal-image pool and native-resolution test annotations. `code/study.py` defines the methods, metrics, and seeds used in the manuscript tables.

## Environment

The experiment environment is Python 3.12, PyTorch 2.8/CUDA 12.8, and an NVIDIA RTX 4090. Use a dedicated environment:

```bash
pip install -r requirements.txt
python -m pytest -q code/test_model.py code/test_metrics.py code/test_residual.py code/test_pipeline.py
```

The scripts download public DINOv2-S/14 register-token and Wide-ResNet-50-2 weights to `pretrained/`. Full training and batch evaluation require a CUDA GPU. Single-image directional inference also supports CPU. Frozen feature caches use approximately 20 GB for the two datasets in addition to prepared images and checkpoints.

## Data

```bash
python code/prepare_data.py 3cad
python code/download_ad2.py
python code/prepare_data.py ad2
python code/audit_data.py 3cad mvtec_ad2
python code/cache_features.py 3cad
python code/cache_features.py mvtec_ad2
```

An existing 3CAD archive can be supplied with `python code/prepare_data.py 3cad --archive /path/to/3cad.zip`.

The AD2 downloader pins the public mirror `pjramg/MVTecAD2_FO` to revision `9a6d209dc83082a4ab0761c4792de957f56b053f`. `HF_ENDPOINT` can select another Hugging Face endpoint. The reported partition is `test_public`. MVTec AD 2 is distributed under CC BY-NC-SA 4.0; dataset terms remain applicable to downloaded data.

The 3CAD protocol excludes 52 training-side byte duplicates of test images. A content-hash split produces 9,406 optimization and 1,035 normal-validation images, with all 16,546 official test images retained. AD2 uses its 2,528 training, 302 validation, and 1,084 public-test images. Every comparison uses these same partitions. Category labels are used for reporting, not as model inputs.

Data preparation records source SHA256 hashes, preserves native masks, and caps RGB images at a longest side of 1024 pixels before model resizing. AD2 conversion receipts support resumption after raw downloads have been removed.

## Training

```bash
python code/train.py --dataset 3cad --variant denoising_vmf_p25 --seed 17
python code/train.py --dataset mvtec_ad2 --variant denoising_vmf_p25 --seed 17
```

Training uses 280-pixel inputs, batch size 16, and 10,000 updates. Repeat with seeds 29 and 43 for the main method and matched deterministic baseline. The final checkpoint is evaluated. Checkpoints retain optimizer and random-generator state during training so interrupted runs can resume with the same batch order.

| Variant | Reconstruction | Directional perturbation |
| --- | --- | --- |
| `dinomaly` | Dinomaly cosine objective | None |
| `vmf_single` | Single vMF per hierarchy | None |
| `denoising_cosine_p25` | Dinomaly cosine objective | Probability 0.25 per image |
| `denoising_vmf_p25` | Single vMF per hierarchy | Probability 0.25 per image |
| `vmf_mixture` | Independent uniform three-component mixtures | None |
| `context_mixture` | Independent contextual mixtures | None |
| `coupled_mixture` | Contextual mixture with one shared component | None |

The full-probability variants `denoising_cosine` and `denoising_vmf` reproduce the initial normal-validation comparison. `code/validate_denoising.py` evaluates clean and corrupted normal validation features. `--p25` evaluates reduced-probability variants after that report exists. The development gate requires at most 10% clean-error growth and a reduction in corrupted-region error on both datasets. The complete queue in `code/pipeline.py` reproduces these normal-only decisions, all scheduled comparisons, accepted denoising repeats, score ablations, and direct transfer. It uses one GPU job and at most two CPU metric jobs concurrently. Training also takes a per-run file lock before loading or saving a checkpoint, so duplicate launches wait and then reuse a completed run. Selected feature splits are preloaded in host RAM and training workers prefetch pinned batches.

```bash
python code/baselines.py --dataset 3cad --variant rd --seed 17
python code/baselines.py --dataset 3cad --variant rdpp --seed 17
python code/patchcore_runner.py --dataset 3cad --seed 17
python code/pipeline.py
```

RD and RD++ retain their 256-pixel inputs, batch size 16, and original learning rates. RD++ uses the upstream seeded Simplex noise. Its prepared training images are cached in RAM, and data loading runs in the main process for compatibility with Numba's OpenMP runtime. PatchCore uses 100,000 uniformly sampled candidate patches and a 10,000-descriptor approximate greedy coreset. Its exhaustive GPU squared-L2 search disables TF32 and was checked against the official FAISS CPU implementation.

## Evaluation and Transfer

```bash
python code/evaluate.py --run 3cad_denoising_vmf_p25_s17 --target 3cad
python code/evaluate.py --run 3cad_denoising_vmf_p25_s17 --target mvtec_ad2
python code/evaluate.py --run mvtec_ad2_denoising_vmf_p25_s17 --target 3cad
```

Transfer uses the source checkpoint directly, without target training or target-specific score normalization. Metrics are I-AUROC, P-AUROC, P-AP, and AUPRO, computed per category and macro-averaged. AUPRO integrates to FPR 0.30 on 3CAD and 0.05 on AD2. Pixel AP uses stepwise recall integration. Pixel statistics use 262,144 histogram bins and connected-region weighting checked against ADEval.

Stored prediction maps are low resolution. Evaluation applies bilinear resizing, Gaussian smoothing with sigma 4 at the model-input resolution, and resizing to native annotation dimensions. Reconstruction methods use the maximum smoothed value for image scoring. PatchCore retains the maximum patch distance.

Apply a trained directional checkpoint to one image:

```bash
python code/infer.py --checkpoint results/3cad_denoising_vmf_p25_s17/model.pt --image /path/to/image.png --output results/example
```

The output contains a native-resolution anomaly map, token map, and image score. Use `--device cpu` for CPU inference. `--score angular_standardized` and `--score angular_deviation` expose the stored single-vMF score alternatives. The default score is the vMF negative log-likelihood ratio. These alternatives are distinct scores, not calibrated probabilities.

## Results and Figures

```bash
python code/audit_experiments.py
python code/research_results.py
python code/make_tables.py
python code/plot_architecture.py
python code/plot_results.py
```

The audit validates checkpoint hashes, split identifiers, native pixel counts, and metric aggregation, and exports `results/measurements.csv`. The research export retains every completed measured variant, including exploratory variants outside the manuscript tables. Tables are generated only after their required measurements exist. Figures are exported as PDF, SVG, and PNG; architecture and perturbation figures use Matplotlib, with the perturbation example drawn from normal training features. The optional tangent-residual scorer and its tests are retained in `code/residual_rd.py` for reproducing the associated reconstruction-score comparison.

## Numerical Details

The vMF partition uses float64 exponentially scaled Bessel functions with an analytical derivative. Concentrations lie in `[0.25d, 16d]`. All reconstruction variants use the same positive linear-attention kernel accumulated in float32. This avoids subtractive cancellation in the equivalent ELU-plus-one expression.

Denoising uses two square regions per selected image, each one quarter of the token-grid side. The regions share their support across hierarchies. Directions are replaced by the spherical midpoint with the previous normal image in the batch, retaining each original token norm. Prefix tokens remain unchanged. An antipodal pair keeps the original direction.

## Upstream Code

- [Dinomaly](https://github.com/guojiajeremy/Dinomaly), revision `a61959d3d0e53aa8f3002f7788427aa27d4ecdd2`, Apache-2.0.
- [RD4AD](https://github.com/hq-deng/RD4AD), revision `6554076872c65f8784f6ece8cfb39ce77e1aee12`, MIT.
- [RD++](https://github.com/tientrandinh/Revisiting-Reverse-Distillation), revision `7f2ceb7c87e602617b8600e1a498f7ef7f5247d6`, MIT.
- [PatchCore](https://github.com/amazon-science/patchcore-inspection), revision `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`, Apache-2.0.

Third-party modules retain their licenses. Cite the respective original papers when using their methods. Data, model weights, predictions, logs, and manuscript files are excluded from version control.
