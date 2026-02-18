## Run interactive
- `source scripts/setup_python.sh`
- `julia scripts/train_tnp.jl 64 100_000` # train TNP
- `julia scripts/run_bosip.jl 64 100_000` # use trained TNP in BOSIP

## Run script
- `julia`
- Edit the settings in "scripts/queue_job.jl"
- `include("scripts/queue_job.jl")`
- `queue_job(args...)`: `args` are passed to the script
