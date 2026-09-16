import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from filelock import FileLock

from model import DirectionalRD
from optimizers import StableAdamW

ROOT = Path(__file__).resolve().parent.parent


class CachedFeatures(Dataset):
    def __init__(self, dataset, size, split):
        folder = ROOT / "data" / dataset
        self.rows = json.loads((folder / "manifest.json").read_text())
        cache = folder / f"features_{size}"
        assert (cache / "COMPLETE").exists(), "Feature extraction is incomplete"
        self.meta = json.loads((cache / "metadata.json").read_text())
        assert self.meta['manifest_sha256'] == hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(), 'Feature cache and manifest differ'
        assert self.meta['shape'][0] == len(self.rows)
        self.features = np.memmap(cache / "features.dat", mode="r", dtype=np.float16, shape=tuple(self.meta["shape"]))
        has_validation = any(row["split"] == "validation" for row in self.rows)
        test_hashes = {row['source_sha256'] for row in self.rows if row['split'] in {'test', 'test_public'}}
        self.indices = []
        for index, row in enumerate(self.rows):
            held_out = int(hashlib.sha256(("normal-validation:" + row["source_sha256"]).encode()).hexdigest()[:8], 16) % 10 == 0
            eligible = row['source_sha256'] not in test_hashes
            if split == "train":
                selected = row["split"] == "train" and eligible and (has_validation or not held_out)
            elif split == "validation":
                selected = eligible and (row["split"] == "validation" if has_validation else row["split"] == "train" and held_out)
            else:
                selected = row["split"] in {"test", "test_public"}
            if selected:
                self.indices.append(index)
        if split != "test":
            assert all(self.rows[index]["label"] == 0 for index in self.indices)
        # The feature cache is much smaller than available host memory. Keep
        # the selected split resident so workers do not repeatedly fault random
        # memmap pages while the GPU waits for the next batch.
        self.preloaded = np.ascontiguousarray(self.features[self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        if self.preloaded is not None:
            return torch.from_numpy(self.preloaded[position]), index
        return torch.from_numpy(self.features[index].copy()), index


class FixedStepBatches(Sampler):
    """Epoch permutations depend only on the seed, including after resume."""

    def __init__(self, count, batch, seed, start, stop):
        self.count, self.batch, self.seed = count, batch, seed
        self.start, self.stop = start, stop
        self.per_epoch = count // batch
        assert self.per_epoch > 0

    def __iter__(self):
        previous = None
        for step in range(self.start, self.stop):
            epoch, position = divmod(step, self.per_epoch)
            if epoch != previous:
                permutation = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                previous = epoch
            yield permutation[position * self.batch:(position + 1) * self.batch]

    def __len__(self):
        return self.stop - self.start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["3cad", "mvtec_ad2"])
    parser.add_argument("--variant", choices=DirectionalRD.VARIANTS, default="denoising_vmf_p25")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    locks = ROOT / 'logs' / 'run_locks'
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / f'{args.dataset}_{args.variant}_s{args.seed}.lock')):
        train_run(args)


def train_run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    run = ROOT / "results" / f"{args.dataset}_{args.variant}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "COMPLETE").exists():
        print("Already complete", run, flush=True)
        return
    train = CachedFeatures(args.dataset, args.size, "train")
    validation = CachedFeatures(args.dataset, args.size, "validation")
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    splits = {"train": [train.rows[i]["id"] for i in train.indices],
              "validation": [validation.rows[i]["id"] for i in validation.indices]}
    assert not set(splits["train"]) & set(splits["validation"])
    (run / "split_ids.json").write_text(json.dumps(splits))
    val_loader = DataLoader(validation, batch_size=args.batch, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True, prefetch_factor=4,
                            generator=torch.Generator().manual_seed(args.seed))
    model = DirectionalRD(args.variant).cuda()
    optimizer = StableAdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-10)
    step = 0
    if (run / "last.pt").exists():
        checkpoint = torch.load(run / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    loader = DataLoader(train, batch_sampler=FixedStepBatches(len(train), args.batch, args.seed, step, args.steps),
                        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                        generator=torch.Generator().manual_seed(args.seed))
    print("Training", vars(args), "normal images", len(train), "validation", len(validation),
          "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    timer = time.time()
    interval_losses = []
    while step < args.steps:
        for features, _ in loader:
            if step >= args.steps:
                break
            model.train()
            features = features.cuda(non_blocking=True).float()
            rate = 2e-3 * (step + 1) / 100 if step < 100 else 2e-4 + 0.5 * (2e-3 - 2e-4) * (1 + math.cos(math.pi * (step - 100) / (args.steps - 100)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(features)
                loss = model.loss(features, output, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            interval_losses.append(float(loss.detach()))
            if step % 100 == 0:
                record = {"step": step, "loss": float(np.mean(interval_losses)), "lr": rate,
                          "gradient_norm": float(grad), "elapsed_seconds": time.time() - timer}
                with (run / "training.jsonl").open("a") as log:
                    log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                interval_losses = []
            if step % 1000 == 0 or step == args.steps:
                model.eval()
                values = []
                with torch.inference_mode():
                    for batch, _ in val_loader:
                        batch = batch.cuda().float()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(batch)
                            value = model.loss(batch, output, step)
                        values.extend([float(value)] * len(batch))
                with (run / "validation.jsonl").open("a") as log:
                    log.write(json.dumps({"step": step, "normal_loss": float(np.mean(values))}) + "\n")
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                              "config": vars(args), "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                              "numpy_rng": np.random.get_state()}
                torch.save(checkpoint, run / "last.pt.tmp")
                (run / "last.pt.tmp").replace(run / "last.pt")
    torch.save({"model": model.state_dict(), "config": vars(args), "step": step}, run / "model.pt")
    (run / "COMPLETE").write_text(str(step))
    (run / "last.pt").unlink(missing_ok=True)
    print("Training complete", run, flush=True)


if __name__ == "__main__":
    main()
