## Run interactive
- `source scripts/setup_python.sh`
- `julia scripts/train_tnp.jl 64 100_000` # train TNP
- `julia scripts/run_bosip.jl 64 100_000` # use trained TNP in BOSIP

## Run script
- `julia`
- Edit the settings in "scripts/queue_job.jl"
- `include("scripts/queue_job.jl")`
- `queue_job(args...)`: `args` are passed to the script

## Models Nomenclature
- "": original from the paper
- "v1": attention-weighted mean, mlp mean residuals, mlp log-std, rho-gate, rho and a0 into mlp
- "lhc-sampler": context data are partially uniform and partially clustered as a Gaussian mixture, target data are lhc
