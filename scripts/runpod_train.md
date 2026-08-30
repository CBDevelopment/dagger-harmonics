# Training DAGGER on a Runpod GPU pod

A record of how this branch's training pod was actually stood up — provisioning,
data transfer, a lean dependency install, and the real gotchas hit along the way.
Kept as a runbook: follow it top to bottom for a fresh pod, or skip to
[Gotchas](#gotchas-hit-this-session) if you're debugging a repeat.

**In a hurry?** [Quickstart](#quickstart) below has copy-paste commands for the
whole start → train → view results → shut down cycle. Everything after that is
the detailed walkthrough and the reasons behind each step, for when something
doesn't just work.

## Quickstart

Five steps, assuming you already have a network volume from a prior run (skip
to [step 1 below](#1-pick-a-data-center-with-actual-gpu-stock--dont-assume) if
starting completely fresh — no volume yet). Fill in `<vol-id>` and the SSH
`<key>`/`<port>`/`<ip>` from your own session as you go; each command below
tells you where they come from.

### 1. Start a pod

```bash
export MSYS_NO_PATHCONV=1   # Windows/Git Bash only -- see Gotchas if you skip this
runpodctl gpu list          # confirm current stock before picking a GPU/DC
runpodctl pod create --name dagger-train \
  --template-id runpod-torch-v280 --gpu-id "<GPU name from gpu list>" \
  --cloud-type SECURE \
  --data-center-ids <DC the volume is pinned to> \
  --network-volume-id <vol-id> --volume-mount-path /workspace \
  --ssh --terminate-after <iso8601, well past the run>

runpodctl pod get <pod-id>   # poll until ssh.ip / ssh.port appear
```

Set up the environment (first time on a *new* pod only — skip if reusing a pod
that's already set up):

```bash
ssh -i <key> -p <port> root@<ip> \
  "git clone --branch main https://github.com/CBDevelopment/dagger-harmonics.git /workspace/dagger-harmonics"

# One clean install, no torch surprises: torch-harmonics/torchmetrics must NOT
# pull their own deps (that's what silently upgrades the pre-baked torch --
# see Gotchas); everything else is safe to install normally.
ssh -i <key> -p <port> root@<ip> \
  "pip install --break-system-packages --no-input --no-deps 'torch-harmonics==0.8.0' torchmetrics && \
   pip install --break-system-packages --no-input lightning-utilities tqdm pydantic-settings python-dotenv scipy matplotlib pandas"

ssh -i <key> -p <port> root@<ip> "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
# should print the template's original torch version (e.g. 2.8.0+cu128) and True
```

If `/workspace/data/val_data_2010.p` isn't already on the volume from a prior
run, transfer it now (~1.7GB, takes a few minutes):

```bash
ssh -i <key> -p <port> root@<ip> "mkdir -p /workspace/data"
scp -i <key> -P <port> data/val_data_2010.p data/omni_features.p \
  data/supermag_features.p data/scalers.p root@<ip>:/workspace/data/
```

### 2. Run training

```bash
ssh -i <key> -p <port> root@<ip> 'cat > /workspace/run_train.py' <<'PY'
import sys
sys.path.insert(0, "/workspace/dagger-harmonics/src")
from dagger_harmonics.train import train

train(
    n_epochs=100,
    lr=5e-3,
    weight_decay=5e-5,
    batch_size=256,        # NOT the paper's 8500 -- see Gotchas, that starves
                            # this dataset's size down to ~6 steps/epoch
    patience=15,
    max_records=None,      # full dataset; use e.g. 2000 for a quick smoke test
    save_path="/workspace/outputs/dagger_model.pt",
    device="cuda",
)
PY

ssh -i <key> -p <port> root@<ip> '
  export DAGGER_DATA_PATH=/workspace/data
  export MPLBACKEND=Agg
  cd /workspace/dagger-harmonics
  setsid env PYTHONPATH=src python3 /workspace/run_train.py > /workspace/train.log 2>&1 </dev/null &
  echo LAUNCHED
'
```

`git pull` first inside `/workspace/dagger-harmonics` on the pod if you're
reusing one from an earlier session, to pick up any code changes. Confirm it's
actually running (the launch command above returns immediately, before
training finishes):

```bash
ssh -i <key> -p <port> root@<ip> "ps aux | grep run_train.py | grep -v grep"
```

With batching, a full epoch now takes **seconds**, not minutes — a run
typically finishes or early-stops within a few minutes, not hours.

### 3. View training progress and results

Runpod's own web console **cannot** show this training's output — it only
streams the container's entrypoint process, and this runs detached over SSH
as a separate process. Two ways to actually see it:

```bash
# live, from your own machine (Ctrl+C to stop watching -- training keeps running)
ssh -i <key> -p <port> root@<ip> "tail -f /workspace/train.log"
```

or open **Runpod console → your pod → Connect → Start Web Terminal** for a
browser shell into the pod, then run the same `tail -f` there.

Once it's finished (`ps aux | grep run_train.py` shows nothing), pull the
results down to look at locally:

```bash
scp -i <key> -P <port> \
  root@<ip>:/workspace/outputs/dagger_model.pt \
  root@<ip>:/workspace/outputs/dagger_model.history.json \
  root@<ip>:/workspace/outputs/dagger_model.png \
  outputs/trained_models/
```

- **`dagger_model.png`** — the loss curve (train/val MAE, val RMSE per epoch).
  Open it directly, or reload `dagger_model.history.json` to re-plot/analyze
  it yourself:
  ```python
  import json
  hist = json.load(open("outputs/trained_models/dagger_model.history.json"))
  hist["train"]  # list of {"mae": ..., "mse": ..., "rmse": ...}, one per epoch
  hist["val"]    # same shape
  ```
- **`dagger_model.pt`** — the checkpoint (best epoch's weights, by val MAE).
  Load with `DAGGER(input_size=...).load_state_dict(torch.load(path, weights_only=True))`.

Both the checkpoint and history are written **every epoch**, not just at the
end — if a run gets interrupted, whatever's on `/workspace/outputs/` is still
usable, not lost.

### 4. Shut down (stop paying for the GPU)

```bash
runpodctl pod remove <pod-id>          # stops billing immediately
runpodctl network-volume list          # confirm the volume (data + code +
                                        # checkpoints) is still there
```

The **pod** is what costs real money per hour and should be removed once
you're done for the session. The **network volume** only costs a small
storage fee and is worth keeping between sessions — it already has the
dataset and cloned repo on it, so the next run skips the data transfer
entirely. Delete it only when you're fully done with this project's Runpod
work:

```bash
runpodctl network-volume delete <vol-id>
```

---

## Prerequisites

- `RUNPOD_API_KEY` resolvable by `runpodctl` — either `flash login` (browser OAuth,
  saves to `~/.runpod/config.toml`) or `runpodctl config --apiKey <key>` directly.
  **Note:** on at least one Windows setup, `flash login` saved a key that
  `runpodctl` still didn't pick up (a `runpodctl user` right after kept returning
  `no_credentials`) — `runpodctl config --apiKey <key>` is the reliable fix
  regardless of what `flash login` already did.
- `runpodctl` installed (`curl -sSL https://cli.runpod.net | bash`, or see
  `runpod-usage/reference/getting-started.md` in the Runpod skill).

## 1. Pick a data center with actual GPU stock — don't assume

`runpodctl gpu list` reports live per-datacenter stock. Availability is volatile
and DC-specific: a datacenter can show zero stock for *every* GPU type at all
(this happened with `US-IL-1` mid-session), and not every datacenter supports
network volumes at all (`runpodctl network-volume create` will name the valid
set in its error if you pick a bad one). Check both before committing:

```bash
runpodctl gpu list                    # per-DC stock + community/secure pricing
runpodctl network-volume create --name x --size 20 --data-center-id <dc>
# ^ if <dc> doesn't support volumes, the error lists the ones that do
```

What actually worked this session: `EU-RO-1`, GPU id `NVIDIA RTX PRO 4500
Blackwell` (32GB, secure-cloud only, ~$0.72/hr, HIGH stock at the time) — found
after `US-IL-1` (originally planned) turned out to have zero stock of anything,
and `US-NC-1` (which *did* have RTX 4090 stock) turned out not to support
network volumes at all.

## 2. Create the network volume (holds data + code + checkpoints, survives the pod)

```bash
runpodctl network-volume create --name dagger-train --size 20 --data-center-id EU-RO-1
# → volume id, pinned to EU-RO-1 -- the pod below MUST run in the same DC
```

## 3. Provision the pod

No `--ports`: this is a batch job, not a server.

```bash
runpodctl pod create --name dagger-train \
  --template-id runpod-torch-v280 --gpu-id "NVIDIA RTX PRO 4500 Blackwell" \
  --cloud-type SECURE \
  --data-center-ids EU-RO-1 \
  --network-volume-id <vol-id> --volume-mount-path /workspace \
  --ssh --terminate-after <iso8601, well past the run>
```

Poll `runpodctl pod get <pod-id>` until `ssh.ip`/`ssh.port` appear (or add
`--wait --wait-timeout 8m` to block until then). Read the ssh key path from
that same response.

## 4. Transfer the data (1.7GB `val_data_2010.p` + 3 small sidecars)

Plain `scp` over the SSH connection from step 3 — no need for `runpodctl
send`/`receive` since SSH access is already there:

```bash
ssh -i <key> -p <port> root@<ip> "mkdir -p /workspace/data"
scp -i <key> -P <port> data/val_data_2010.p data/omni_features.p \
  data/supermag_features.p data/scalers.p root@<ip>:/workspace/data/
```

## 5. Install the repo — lean deps, reusing the template's system torch

The repo is public; clone with no credentials. Then the dependency install is
the part worth doing carefully — see [Gotchas](#gotchas-hit-this-session) for
why the naive version wastes ~7-8GB.

```bash
ssh -i <key> -p <port> root@<ip> \
  "git clone --branch <branch> https://github.com/CBDevelopment/dagger-harmonics.git /workspace/dagger-harmonics"
```

**Check what Python the template's torch actually lives in first** — it may not
be the same interpreter `uv` would pick:

```bash
ssh ... "python3 -c 'import torch; print(torch.__version__, torch.__file__)'"
```

If it matches the project's `requires-python` (it didn't here — template had
3.12, project requires >=3.13), skip `uv`/venvs entirely and install straight
into the system interpreter, matching the golden-path convention for reusing a
template's pre-baked torch:

```bash
ssh ... "pip install --break-system-packages --no-input --no-deps 'torch-harmonics==0.8.0' torchmetrics"
ssh ... "pip install --break-system-packages --no-input lightning-utilities tqdm pydantic-settings python-dotenv scipy matplotlib pandas"
```

**Two separate calls, not one** — only `torch-harmonics`/`torchmetrics` need
`--no-deps` (they're the ones that would otherwise drag in a fresh torch;
see the gotcha below). Applying `--no-deps` to *everything* in one call (an
earlier version of this doc did) leaves `torchmetrics` missing
`lightning-utilities` and `matplotlib`/`pandas` missing their own harmless
sub-dependencies (`cycler`, `fonttools`, `pydantic`, etc.) — none of those
touch torch, so they're safe to install normally in a second call.

**`--no-deps` is not optional here** — see the torch-upgrade gotcha below.
`numpy` is usually already present in ML-focused templates; check before
installing it too.

Run training with `PYTHONPATH=/workspace/dagger-harmonics/src python3 ...` —
no venv, no `uv run` (see gotcha below for why).

## 6. Launch training, detached, logging to the volume

```bash
ssh -i <key> -p <port> root@<ip> 'cat > /workspace/run_train.py' <<'PY'
import sys
sys.path.insert(0, "/workspace/dagger-harmonics/src")
from dagger_harmonics.train import train

train(
    n_epochs=100,
    lr=5e-3,
    weight_decay=5e-5,
    batch_size=256,           # see the batch_size gotcha below -- NOT the
                               # paper's 8500 for a dataset this size
    patience=15,
    max_records=None,         # full dataset; set e.g. 2000 for a smoke test first
    save_path="/workspace/outputs/dagger_model.pt",
    device="cuda",
)
PY

ssh -i <key> -p <port> root@<ip> '
  export DAGGER_DATA_PATH=/workspace/data
  export MPLBACKEND=Agg
  setsid env PYTHONPATH=/workspace/dagger-harmonics/src python3 /workspace/run_train.py \
    > /workspace/train.log 2>&1 </dev/null &
  echo LAUNCHED
'
```

`MPLBACKEND=Agg` matters: `train()` ends by saving a loss-curve PNG via
`fig.savefig()`, which needs a non-interactive backend on a headless pod (the
`plt.show()` call alongside it is a harmless no-op there).

**Always smoke-test first**: `max_records=2000, n_epochs=2` runs in well under a
minute on a modern GPU and confirms the data path, device, and save path are
all correct before committing to a longer run. With batching, even a full run
now typically finishes in minutes, not hours — see the batch_size gotcha
below for why the number matters more than "bigger is faster."

## 7. Monitor (tail the log — there's no URL to poll)

```bash
ssh -i <key> -p <port> root@<ip> 'tail -n 40 /workspace/train.log'
ssh -i <key> -p <port> root@<ip> 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

Success looks like falling `train_mae`/`val_mae` lines each epoch and a final
`Model saved -> /workspace/outputs/dagger_model.pt`. (The tqdm progress bar's
Unicode block characters render as mangled bytes through some SSH/console
paths — harmless, ignore them; the printed metric lines are plain ASCII.)

Runpod's web console **cannot** show this log — it only streams the
container's entrypoint process, and training runs detached over SSH as a
separate process. `tail -f` over SSH (as above) or a Runpod web terminal are
the only ways to watch it live.

## 8. Retrieve the results, then free the GPU

Every epoch (not just the last one) writes a checkpoint, a history JSON, and
(at the end) a loss-curve PNG — so even an interrupted run leaves something
usable on `/workspace/outputs/`:

```bash
scp -i <key> -P <port> \
  root@<ip>:/workspace/outputs/dagger_model.pt \
  root@<ip>:/workspace/outputs/dagger_model.history.json \
  root@<ip>:/workspace/outputs/dagger_model.png \
  outputs/trained_models/

runpodctl pod remove <pod-id>            # stop paying for the GPU
runpodctl network-volume list            # data + code + checkpoints persist here
```

---

## Gotchas hit this session

### Git Bash on Windows silently corrupts POSIX-looking arguments

Running `runpodctl pod create ... --volume-mount-path /workspace` from Git Bash
on Windows produced a pod that could **never start** — its container kept
failing with `invalid mount config for type "bind": invalid mount path:
'C:/Program Files/Git/workspace' mount path must be absolute`. MSYS (Git Bash's
POSIX layer) auto-converts any argument that *looks* like a POSIX path into a
Windows path before handing it to a plain Windows binary like `runpodctl` — so
`/workspace` silently became `C:/Program Files/Git/workspace`, and that
corrupted value is what got sent to Runpod's API.

**Symptom to watch for**: a pod stuck at `runtimeStatus: initializing` /
`awaiting_container` indefinitely, SSH never comes up. `runpodctl pod get
<id>` will show the mangled `volumeMountPath` if this is the cause.

**Fix**: set `MSYS_NO_PATHCONV=1` for any `runpodctl`/similar command that
takes a `/`-leading argument:

```bash
export MSYS_NO_PATHCONV=1
runpodctl pod create ... --volume-mount-path /workspace ...
```

### `uv sync`'s `--no-install-package torch` doesn't skip torch's own CUDA deps

The goal was reusing the template's pre-installed torch (2.8.0+cu128, ~5GB
already on disk) instead of `uv sync` downloading its own copy. Passing
`--no-install-package torch` only excludes the `torch` wheel itself — `uv`
still resolves and installs every package torch's *own* metadata depends on
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-nccl-cu12`, `triton`, ~15
packages, several GB). Had to exclude all of them individually
(`--no-install-package nvidia-cublas-cu12 --no-install-package ...`) to
actually get the savings.

### `uv venv --system-site-packages` only helps if the Python versions match

Even after excluding the CUDA packages, the new venv still couldn't `import
torch` — `uv venv` had picked Python 3.13 (the project's
`requires-python = ">=3.13"`), but the template's torch was installed for
system Python **3.12**. Different interpreter versions have incompatible
binary extension ABIs, so `--system-site-packages` can't bridge them; a
`--system-site-packages` venv only actually sees the system packages when its
Python version matches the one they were installed for.

**Resolution**: rather than loosening the project's `requires-python` just for
one pod, skipped the venv entirely and installed directly into the template's
system Python 3.12 (`pip install --break-system-packages ...`) — the same
pattern the Runpod golden paths already recommend for reusing a template's
torch.

### An unpinned, transitive `torch-harmonics` install silently upgraded torch

First attempt: `pip install --break-system-packages torch-harmonics
torchmetrics ...` with no version pin and no `--no-deps`. `torch-harmonics`
resolved to its latest release (0.9.2, newer than this project's pinned
`>=0.8.0`), which requires a newer torch — pip **uninstalled** the perfectly
good pre-baked `torch==2.8.0+cu128` and replaced it with a freshly-downloaded
`torch==2.11.0+cu130`, also breaking the template's `torchvision`/`torchaudio`
version pins (unused by this project, so harmless in practice, but wasted the
entire "reuse the template's torch" exercise for that install and pulled a
fresh ~5GB CUDA 13 stack).

**Fix going forward**: always pin the exact version and pass `--no-deps` when
the point is to *not* let pip touch an already-satisfied dependency:

```bash
pip install --break-system-packages --no-deps torch-harmonics==0.8.0
```

The session recovered by accepting the new (working, CUDA-confirmed) torch
2.11.0 rather than trying to force the exact original version back — not worth
the risk of reconstructing an unknown-provenance pre-baked wheel.

### `uv run` re-triggers its own full sync, ignoring prior manual `uv sync` flags

After successfully building a lean, `--no-install-package`-flagged venv (down
to 1.4GB), running `uv run --project ... python -c "..."` silently kicked off
*another* full sync with none of those exclusions — the venv ballooned back to
6.8GB before being caught and killed. `uv run` always re-syncs to match the
lockfile unless told not to. Once bypassing the venv entirely in favor of
system Python (see above), this stopped being relevant — just don't mix
`uv run` with a hand-tuned dependency set; use `python3` / the venv's own
`bin/python` directly.

### Container disk vs. network volume

The pod's container disk (`--container-disk-in-gb`, default 20GB) is separate
from the attached network volume and is wiped on pod stop/restart. All the
`pip install` work above landed on the *container* disk (`/usr/local/lib/...`),
not `/workspace` — worth watching with `df -h /` if installing much there,
since it's a fixed, smaller budget than the volume.

### `_sh_basis` was CPU-bound the whole time, not the GPU's fault

Early runs showed 2% GPU utilization and ~990% CPU during training — the
spherical-harmonic basis evaluation ran through `scipy.special.lpmv` on CPU
numpy arrays regardless of `device`. This got fixed in the codebase itself
(pure-PyTorch tensor recurrence, verified against scipy as ground truth) —
nothing to work around here anymore, but if you see the GPU sitting near-idle
during training on an older checkout, this is why.

### `batch_size` must be sized to *this* dataset, not copied from the paper

The paper's Table 2 specifies `batch_size=8500` — but that was against ~10
years of 1-minute-cadence data (millions of records). Against this project's
~46k training records, `batch_size=8500` leaves only **~6 optimizer steps per
epoch**, nowhere near enough for real convergence: `train_mae` barely moved
before `patience` exhausted itself. Batching itself is a huge win regardless
(unbatched: ~68 records/sec; `batch_size=256`: ~15,000 records/sec;
`batch_size=8500`: ~83,000 records/sec, all benchmarked on one RTX PRO 4500)
— the number just needs to leave enough steps per epoch to actually learn.
`batch_size=256` (~180 steps/epoch against the full dataset) is a reasonable
default; since each epoch is now seconds not minutes, raising `n_epochs` and
`patience` costs almost nothing, so lean generous there instead of maximizing
`batch_size`.

### A stalled/flat loss curve isn't necessarily a bug

After fixing the batch-size issue, `train_mae` still plateaus fast (flat by
epoch 2-3, at every dataset size tried: 500, 2000, 5000, and the full ~46k
records) and val MAE settles around ~5nT without much further improvement.
Checked for an actual bug first — scanned per-batch gradient norms manually
(stayed bounded, 1.6 → ~0.5-0.9, no explosion) — before concluding this
matches the paper's own §4.2: DAGGER is documented to "under-forecast" large
events, plausibly because `weight_decay` constantly shrinks the SH
coefficients against L1 loss's fixed ±1 gradient magnitude. The paper trained
on ~10 years of data for ~40 hours; this dataset is one year. Worth knowing
before assuming a flat curve means something is broken.

### `print()`ing a Unicode arrow (→) can crash on Windows, hiding a successful save

`train()` used to print `f"Model saved → {path}"` after `torch.save(...)`
succeeded. On a Windows terminal using the `cp1252` codepage (not UTF-8), that
`print()` call itself raises `UnicodeEncodeError` — which looks like the
whole training run crashed, when the checkpoint was actually written
successfully just before the print that failed. Fixed in the codebase (plain
ASCII `->` now) — worth remembering if you ever see this exact traceback
after modifying a log message: check whether the file it was about to
announce actually exists before assuming the underlying operation failed.
