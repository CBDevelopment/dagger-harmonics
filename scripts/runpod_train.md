# Training DAGGER on a Runpod GPU pod

A record of how this branch's training pod was actually stood up — provisioning,
data transfer, a lean dependency install, and the real gotchas hit along the way.
Kept as a runbook: follow it top to bottom for a fresh pod, or skip to
[Gotchas](#gotchas-hit-this-session) if you're debugging a repeat.

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
ssh ... "pip install --break-system-packages --no-deps \
  'torch-harmonics==0.8.0' torchmetrics tqdm pydantic-settings python-dotenv scipy matplotlib pandas"
```

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
    n_epochs=20,
    patience=5,
    max_records=None,        # full dataset; set e.g. 2000 for a smoke test first
    save_path="/workspace/outputs/dagger_model.pt",
    device="cuda",
)
PY

ssh -i <key> -p <port> root@<ip> '
  export DAGGER_DATA_PATH=/workspace/data
  export MPLBACKEND=Agg
  setsid python3 /workspace/run_train.py > /workspace/train.log 2>&1 </dev/null &
  echo LAUNCHED
'
```

`MPLBACKEND=Agg` matters: `train()` ends with a `plt.show()` loss plot, which
needs a non-interactive backend on a headless pod.

**Always smoke-test first**: `max_records=2000, n_epochs=2` runs in well under a
minute on a modern GPU and confirms the data path, device, and save path are
all correct before committing to a multi-hour full run.

## 7. Monitor (tail the log — there's no URL to poll)

```bash
ssh -i <key> -p <port> root@<ip> 'tail -n 40 /workspace/train.log'
ssh -i <key> -p <port> root@<ip> 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

Success looks like falling `train_mae`/`val_mae` lines each epoch and a final
`Model saved → /workspace/outputs/dagger_model.pt`. (The tqdm progress bar's
Unicode block characters render as mangled bytes through some SSH/console
paths — harmless, ignore them; the printed metric lines are plain ASCII.)

## 8. Retrieve the model, then free the GPU

```bash
scp -i <key> -P <port> root@<ip>:/workspace/outputs/dagger_model.pt outputs/trained_models/
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
