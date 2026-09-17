# Support-Selective Reverse Distillation

SS-RD keeps the Dinomaly-style reconstruction network and changes its normal-image refinement target: `Y = M F(x) + (1-M)F(x_tilde)`. The altered image supplies the decoder input. Clean teacher features supervise the altered support; observed teacher features supervise its complement. Inference uses the original teacher--decoder cosine discrepancy.

This code covers 3CAD, the public test partition of MVTec AD 2, matched target ablations, direct transfer, and the RD, RD++, HFSA, and CFRG comparisons. CFRG is evaluated on 3CAD. Data, trained weights, predictions, and manuscripts are excluded.

## Environment

Measured environment: Python 3.12, PyTorch 2.8.0/CUDA 12.8, RTX 4090. Install a matching PyTorch wheel, then:

    pip install -r requirements.txt
    python code/fetch_hfsa.py
    python code/fetch_cfrg.py
    python code/prepare_cfrg_textures.py

The frozen DINOv2 ViT-S/14 register teacher and WRN50-2 weights use their original public URLs. Licensed upstream source retains its licence files. HFSA is fetched from the commit and SHA-256 values in `code/third_party/hfsa/SOURCE.json`; its source is not bundled. CFRG is also fetched separately from the pinned commit and SHA-256 records in `code/third_party/cfrg/SOURCE.json`. Its adapter provides the NumPy dtype registry required by imgaug 0.4 without downgrading NumPy.

## Data

Prepare the official 3CAD archive and the recorded public AD 2 mirror revision:

    python code/prepare_data.py 3cad --archive /path/to/3cad.zip
    python code/prepare_data.py ad2
    python code/audit_data.py

Preparation preserves native annotation masks and records each original image's SHA-256, dimensions, split, and source path. RGB derivatives have a maximum side length of 1,024 pixels before the model input resize. The shared split removes 3CAD training files whose original content also occurs in test, then reserves normal validation examples using a fixed hash rule.

| Dataset | Normal training | Normal validation | Test | Categories |
|---|---:|---:|---:|---:|
| 3CAD | 9,406 | 1,035 | 16,546 | 8 |
| MVTec AD 2 | 2,528 | 302 | 1,084 | 8 |

Each dataset uses one model shared across categories. Category labels only group evaluation results.

## Reproduce

`SSRD_CACHE_ROOT` may point to a fast disk or a sufficiently large shared-memory directory. Its default is the repository's `cache/` directory.

    python code/ssrd/cache.py --dataset both --size 448
    python code/ssrd/cache_rgb.py
    python code/reproduce.py

The runner reuses completed outputs. It executes 10,000 ordinary-reconstruction updates and 2,000 updates for normal continuation, complete denoising, and support-selective denoising. Seed 17 uses the recorded individual refinement procedure. Independent initialization seeds 29 and 43 use paired target refinement. The 896 study starts from the seed-17 448 checkpoint and uses 2,000 refinement updates.

The convolutional comparisons use 448 inputs, batches of 16, and 12,000 updates. RD/RD++ retain their original architecture; RD++ retains its projection objective and simplex perturbation. HFSA retains its official teacher, axial bottleneck, decoder, and loss, with axial embeddings sized for 448. CFRG keeps its official recovery, discrepancy, and segmentation branches and its DTD/Perlin synthesis. It follows the released Adam optimizer (learning rate 0.0005, betas 0.5/0.999), with decay by 0.2 at updates 9,600 and 10,800. Its final prescribed checkpoint is used. These are common-data multicategory comparisons; the original papers use different per-category protocols.

For a focused run:

    python code/reproduce.py --datasets 3cad --seeds 17 --skip-external --skip-transfer --skip-resolution --skip-timing

## Evaluation

Outputs live in `results/runs/`. Checkpoints include training configuration. Prediction metadata records checkpoint hashes and settings. Stored maps are float32 token residuals, or the three native residual grids for convolutional models. The RD/RD++ evaluator writes memory-mapped arrays batch by batch to bound host-memory use. CFRG stores its smoothed 448-by-448 float32 maps under `SSRD_CACHE_ROOT` and links them into the evaluation folder. This derived cache uses approximately 12.4 GiB for 3CAD and must remain available while computing or auditing metrics.

All test images, including normal backgrounds, enter evaluation. Metrics are category-macro I-AUROC, P-AUROC, P-AP, and AUPRO. Pixel curves use 262,144 score bins; AP uses stepwise recall integration. AUPRO integrates to FPR 0.30 on 3CAD and 0.05 on AD 2. Transfer applies a source-trained model to the other dataset without adaptation.

The matched decoder implements the Dinomaly reconstruction backbone and is distinct from the convolutional RD 2022 baseline. Resolution comparisons use separate normal-continuation controls. Timing uses a fresh process per model, batch size, and repetition: 50 warm-up batches, 100 timed batches, and three repetitions with alternating method order. With normalized inputs already on the GPU, it includes the teacher, decoder, score-map construction, transfer of scores to the CPU, resizing, Gaussian smoothing, and the image-score maximum. Image decoding, normalization, and input upload occur before the timed path. Real-defect masks are used for evaluation and annotated figures, not model fitting.

## Component and support analyses

After the completed runs, reproduce the fixed 256-image normal-validation diagnostic and category-resolved CFRG cases with:

    python code/ssrd/submission_support_study.py
    python code/ssrd/submission_component_cases.py
    python code/ssrd/submission_resolution_timing.py

The support diagnostic uses 32 hash-selected validation images per 3CAD category, fixed line/patch perturbations, and the final checkpoints for seeds 17, 29, and 43. It does not fit or select models. The component analysis reads all category metrics and uses fixed image identifiers for the complementary natural-defect cases. The resolution timing study measures the completed 896 checkpoint at batches 1 and 16, using three isolated rounds of 50 warm-up and 100 timed batches and Gaussian sigma 8. Generated results stay under `results/submission_case_study/`; they are not tracked in this repository.

## Main files

- `code/ssrd/network.py`: backbone, inherited hard-gradient cosine loss, and anomaly map.
- `code/ssrd/spatial_target.py`: perturbation support and seed-17 refinement.
- `code/ssrd/refine_paired.py`: independent paired repetitions.
- `code/ssrd/stable_optimizer.py`: batched RMS-stabilized AdamW.
- `code/ssrd/evaluate_v2.py`: 448 inference and native-mask evaluation.
- `code/ssrd/spatial_highres.py`: 896 selective and complete-target refinement.
- `code/ssrd/train_cnn.py`, `train_hfsa.py`, and `train_cfrg.py`: external comparisons.
- `code/fetch_cfrg.py` and `prepare_cfrg_textures.py`: pinned CFRG source and verified DTD texture preparation.

## Attribution

Built on RD (CVPR 2022), RD++ (CVPR 2023), Dinomaly (CVPR 2025), DINOv2, HFSA (IEEE TCE, 2025), and CFRG/3CAD (AAAI 2025). DeSTSeg, CDO, and related denoising anomaly detectors provide the closest context for the training intervention. Upstream source records and licence files accompany the relevant code.
