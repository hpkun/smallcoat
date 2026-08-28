@echo off
setlocal EnableExtensions

rem Formal Fig.7 sweep: 5 arrival rates x 3 seeds x 3 models.
set "ROOT=outputs\8.22\formal_load_sweep"
set "TRAIN=%ROOT%\train_logs"
set "CKPT=%ROOT%\checkpoints"

if not exist "%TRAIN%" mkdir "%TRAIN%"
if not exist "%CKPT%" mkdir "%CKPT%"

for %%R in (5 15 25 35 45) do (
  for %%S in (11 42 89) do (
    echo [formal] Proposed lambda=%%R seed=%%S
    python train.py --env training --episodes 500 --steps 50 --batch-size 64 --device auto --seed %%S --arrival-rate %%R --redundancy-mode hybrid --actor-attention --output "%TRAIN%\proposed_lambda_%%R_seed_%%S_train.json" --checkpoint-output "%CKPT%\proposed_lambda_%%R_seed_%%S.pt"
    if errorlevel 1 exit /b 1
    python scripts\evaluate_checkpoint.py --algorithm cmaddpg --checkpoint "%CKPT%\proposed_lambda_%%R_seed_%%S.pt" --env training --episodes 100 --steps 50 --seed %%S --arrival-rate %%R --redundancy-mode hybrid --actor-attention --output "%ROOT%\proposed_lambda_%%R_seed_%%S.json"
    if errorlevel 1 exit /b 1

    echo [formal] CMADDPG lambda=%%R seed=%%S
    python train.py --env training --episodes 500 --steps 50 --batch-size 64 --device auto --seed %%S --arrival-rate %%R --redundancy-mode none --output "%TRAIN%\cmaddpg_lambda_%%R_seed_%%S_train.json" --checkpoint-output "%CKPT%\cmaddpg_lambda_%%R_seed_%%S.pt"
    if errorlevel 1 exit /b 1
    python scripts\evaluate_checkpoint.py --algorithm cmaddpg --checkpoint "%CKPT%\cmaddpg_lambda_%%R_seed_%%S.pt" --env training --episodes 100 --steps 50 --seed %%S --arrival-rate %%R --redundancy-mode none --output "%ROOT%\cmaddpg_lambda_%%R_seed_%%S.json"
    if errorlevel 1 exit /b 1

    echo [formal] CMPPO lambda=%%R seed=%%S
    python scripts\run_cmppo.py --env training --episodes 500 --steps 50 --device auto --seed %%S --arrival-rate %%R --redundancy-mode none --output "%TRAIN%\cmppo_lambda_%%R_seed_%%S_train.json" --checkpoint-output "%CKPT%\cmppo_lambda_%%R_seed_%%S.pt" --progress-interval 25
    if errorlevel 1 exit /b 1
    python scripts\evaluate_checkpoint.py --algorithm cmppo --checkpoint "%CKPT%\cmppo_lambda_%%R_seed_%%S.pt" --env training --episodes 100 --steps 50 --seed %%S --arrival-rate %%R --redundancy-mode none --output "%ROOT%\cmppo_lambda_%%R_seed_%%S.json"
    if errorlevel 1 exit /b 1
  )
)

echo [formal] All training and frozen evaluations completed.
endlocal
