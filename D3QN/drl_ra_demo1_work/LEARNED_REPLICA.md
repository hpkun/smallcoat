# DRL-RA Learned Replica

This variant keeps the existing 108-dimensional SAGIN observation and primary
Dueling Double DQN. A second Dueling Double DQN selects zero to two additional
execution targets autoregressively, or chooses `STOP`. The complete set is
submitted once, so internal selections do not advance time or resample links.

## Train and evaluate

```powershell
python train.py --method drl-ra-learned-replica --seed 0
python evaluate.py --method checkpoint `
  --checkpoint outputs/drl-ra-learned-replica_seed0/model.pt `
  --seeds 0 1 2
```

For a short verification run:

```powershell
python train.py --method drl-ra-learned-replica --seed 0 `
  --set training.episodes=1 `
  --set environment.episode_steps=20 `
  --set training.batch_size=4 `
  --set learned_replica.training.batch_size=4
```

To warm-start and temporarily freeze the primary policy, set
`learned_replica.primary_checkpoint` and a positive
`learned_replica.freeze_primary_episodes`. The default is the from-scratch
joint-training control (`primary_checkpoint: null`, freeze duration zero).

## DRL-RA teacher and Replica BC initialization

The primary D3QN remains unchanged. To initialize only the Replica D3QN from
the original DRL-RA heuristic, generate demonstrations on a fixed trace:

```powershell
python generate_teacher_data.py `
  --trace outputs/traces/tasks2000_seed42.json `
  --teacher-checkpoint outputs/seed42_500ep_fixed2000_20261007/drl-ra_seed42/model.pt `
  --output outputs/teacher_data_seed42.json `
  --device cpu --torch-threads 1 --progress
```

Pretrain the replica network with masked cross-entropy, then point
`learned_replica.bc_checkpoint` at the resulting checkpoint:

```powershell
python pretrain_replica_bc.py `
  --data outputs/teacher_data_seed42.json `
  --output outputs/replica_bc_seed42/model.pt `
  --config configs/paper.yaml --epochs 10 --batch-size 128 `
  --device cpu --torch-threads 1

python train.py --method drl-ra-learned-replica --seed 42 `
  --set learned_replica.primary_checkpoint=outputs/seed42_500ep_fixed2000_20261007/drl-ra_seed42/model.pt `
  --set learned_replica.bc_checkpoint=outputs/replica_bc_seed42/model.pt
```

`generate_teacher_data.py` uses the original DRL-RA checkpoint for primary
actions and expands its `_replica_plan(primary)` into backup/STOP labels. It
does not alter the environment or run a reward oracle. BC updates only the
Replica online network and synchronizes its target network; the existing V2
cross-task RL bootstrap then continues unchanged.

## Fine-tune only the BC Replica for 100 episodes

`configs/replica_bc_finetune.yaml` loads the original seed-42 DRL-RA primary
and the independent BC Replica checkpoint. It freezes the primary for all
100 episodes and sets both primary epsilon start/end to zero for greedy
decisions throughout. The primary weights, target network, and Lagrange
multiplier remain fixed. Replica epsilon starts at 0.05, decays by 0.98 per
episode to a floor of 0.01, and its RL learning rate is 1e-5.

```powershell
$runDir = "outputs/seed42_bc_replica_finetune_100ep_20261009"
if (Test-Path $runDir) { throw "Output already exists; choose a new runDir." }
python -u train.py --config configs/replica_bc_finetune.yaml `
  --method drl-ra-learned-replica --seed 42 --device auto `
  --torch-threads 1 --metrics-only --output-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "BC Replica fine-tuning failed." }
```

The training CLI defaults to `--device auto`: CUDA when PyTorch reports it
available, otherwise CPU. The selected device is printed at startup.
Training uses 1000 newly generated tasks per episode; the saved 2000-task
trace remains the evaluation input. Start from the BC checkpoint for this
experiment, rather than the degraded joint-RL checkpoint. If increasing the
episode count, also increase `freeze_primary_episodes` to cover the full run.

## Design boundaries

- Replica state: 108 base features, 20 selected-node bits, normalized count,
  set reliability, reliability gap, and normalized expected energy (132 total).
- Replica actions: 20 execution targets plus `STOP` (21 total), with duplicate
  and unavailable targets masked.
- STOP is masked while set reliability is below the task requirement and
  a feasible backup remains. Once reliable, both STOP and available backups
  remain valid. Three replicas submit automatically. If no backup remains
  below the requirement, forced STOP records `stop_reason="infeasible"`.
  This is a stop condition for the selected set, not the global intrinsic
  feasibility audit. Reliable sets with no backup record
  `no_feasible_candidate_stop`.
- The maximum set size remains three and the old `drl-ra` path is unchanged.
- Replica replay uses `gamma_intra=1.0` within a task's autoregressive
  selection sequence. The terminal internal transition receives the
  task-level constrained reward; between tasks it bootstraps from the next
  task's initial replica state and action mask with `gamma_inter=0.99`.
  Bootstrap is disabled only when the system episode itself ends.
- The environment still uses independent replica-success estimates and its
  original immediate loser-cancellation lifecycle approximation.

The same STOP mask is used for exploration, greedy evaluation, and replay
bootstrap states. Loading an older checkpoint also uses the current mask;
its evaluation behavior therefore changes even though weights still load.
Use a separate output directory when training with the new STOP rule.
