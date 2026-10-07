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

## Design boundaries

- Replica state: 108 base features, 20 selected-node bits, normalized count,
  set reliability, reliability gap, and normalized expected energy (132 total).
- Replica actions: 20 execution targets plus `STOP` (21 total), with duplicate
  and unavailable targets masked.
- The maximum set size remains three and the old `drl-ra` path is unchanged.
- Replica replay uses `gamma_intra=1.0` within a task's autoregressive
  selection sequence. The terminal internal transition receives the
  task-level constrained reward; between tasks it bootstraps from the next
  task's initial replica state and action mask with `gamma_inter=0.99`.
  Bootstrap is disabled only when the system episode itself ends.
- The environment still uses independent replica-success estimates and its
  original immediate loser-cancellation lifecycle approximation.
