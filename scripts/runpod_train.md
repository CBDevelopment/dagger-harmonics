# Training DAGGER on a Runpod GPU pod

A batch-job run (no server, no port) following the shape of Runpod's
`golden-paths/04-finetune-pod` — provision, transfer `val_data_2010.p`, run
`dagger_harmonics.train.train` detached, tail the log, free the GPU.

Prereqs: `RUNPOD_API_KEY` resolvable (`flash login`, or export the key from
https://console.runpod.io/user/settings), `runpodctl` installed, an SSH key
registered (`runpodctl ssh list-keys` / `doctor`).

## 1. Create a network volume (holds data + checkpoints, survives the pod)

```bash
runpodctl network-volume create --name dagger-train --size 20 --data-center-id CA-MTL-1
# pins the volume to CA-MTL-1 -- the pod below MUST run in the same DC.
# swap the DC if you want a different region; check GPU stock there first
# with `runpodctl gpu list --data-center-id <dc>` or list-gpu-types (MCP).
```

## 2. Provision the pod (RTX 4090, community cloud -- $0.34/hr, HIGH stock as of this check)

No `--ports`: this is a batch job, not a server.

```bash
runpodctl pod create --name dagger-train \
  --template-id runpod-torch-v280 --gpu-id "NVIDIA GeForce RTX 4090" \
  --data-center-ids CA-MTL-1 \
  --network-volume-id <vol-id> --volume-mount-path /workspace \
  --ssh --terminate-after <iso8601, well past the run>

runpodctl pod get <pod-id>     # poll until it has a runtime
eval "$(runpodctl ssh info <pod-id> | python3 -c \
  'import sys,json; d=json.load(sys.stdin); print(f"IP={d[\"ip\"]} PORT={d[\"port\"]} KEY={d[\"ssh_key\"][\"path\"]}")')"
```

If the runtime never comes up (`ssh info` stays "pod not ready"), delete and
recreate rather than waiting -- a known bad-machine draw.

## 3. Transfer the data (1.7GB `val_data_2010.p` + 3 small sidecars)

Straight `scp` over the SSH connection from step 2 -- no need for
`runpodctl send`/`receive` since we already have SSH access:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" 'mkdir -p /workspace/data'
scp -i "$KEY" -P "$PORT" \
  data/val_data_2010.p data/omni_features.p data/supermag_features.p data/scalers.p \
  root@"$IP":/workspace/data/
```

## 4. Install the repo on the pod

The repo is public, so a plain clone works with no credentials:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" '
  git clone https://github.com/CBDevelopment/dagger-harmonics.git /workspace/dagger-harmonics &&
  cd /workspace/dagger-harmonics &&
  curl -LsSf https://astral.sh/uv/install.sh | sh &&
  export PATH="$HOME/.local/bin:$PATH" &&
  uv sync
'
```

> If this branch (`worktree-dagger-explainer-doc`) hasn't been merged to
> `main` yet, clone with `--branch worktree-dagger-explainer-doc` instead --
> `main` doesn't have the device-support change yet.

## 5. Launch training, detached, logging to the volume

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" \
  'cat > /workspace/run_train.py' <<'PY'
from dagger_harmonics.train import train

train(
    n_epochs=20,        # paper default; raise if it's still improving
    patience=5,
    max_records=None,   # full ~51k records; set e.g. 2000 for a quick smoke test first
    save_path="/workspace/outputs/dagger_model.pt",
    device="cuda",
)
PY

ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" '
  cd /workspace/dagger-harmonics &&
  export DAGGER_DATA_PATH=/workspace/data &&
  export MPLBACKEND=Agg &&
  export PATH="$HOME/.local/bin:$PATH" &&
  setsid uv run python /workspace/run_train.py > /workspace/train.log 2>&1 </dev/null &
  echo LAUNCHED
'
```

`MPLBACKEND=Agg` matters: `train()` ends with a `plt.show()` loss plot, which
needs a non-interactive backend on a headless pod or it'll error instead of
just silently skipping the display.

**Recommended first run:** smoke-test with `max_records=2000, n_epochs=2`
before committing to a full run -- confirms the data path, device, and save
path are all correct in a couple of minutes rather than finding out after a
long run.

## 6. Monitor (tail the log in separate calls -- there's no URL to poll)

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" 'tail -n 40 /workspace/train.log'
ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" \
  'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

Success looks like falling `train_mae`/`val_mae` lines each epoch and a
final `Model saved → /workspace/outputs/dagger_model.pt`.

## 7. Retrieve the trained model, then free the GPU

```bash
scp -i "$KEY" -P "$PORT" root@"$IP":/workspace/outputs/dagger_model.pt outputs/trained_models/
runpodctl pod remove <pod-id>            # stop paying for the GPU
runpodctl network-volume list            # data + any further checkpoints persist here
runpodctl network-volume delete <vol-id> # once you're done with it
```

## Gotchas

- **Volume and pod must share a data center** -- volumes are DC-locked.
- **`--terminate-after` must exceed the run**, or the pod gets deleted
  mid-training. Use it as a cost guard set generously past your expected
  runtime, not a precise cutoff.
- **Detach with `setsid ... </dev/null &`**, not a bare `&` -- a plain
  background job dies the moment the SSH connection drops.
- **This loop trains one record at a time (batch size 1)** -- `device="cuda"`
  moves compute to the GPU (added in this branch), but with no batching the
  GPU will spend a lot of time on tiny kernel launches. Expect a real but
  moderate speedup over CPU, not a dramatic one, until the training loop is
  batched.
