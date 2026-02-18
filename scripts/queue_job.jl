# ASSUMES pwd == tnp_benchmarks

function queue_job(args...)
    ## Script
    script = "scripts/train_tnp.sh"
    # script = "scripts/run_bosip.sh"

    ## Partition
    partition = "amdgpufast"
    # partition = "amdgpu"

    ## Queue the Job
    args_str = string.(args)
    job_name = split(script, ['/', '.'])[end-1]
    job_name *= "_" * join(args_str, "_")

    cmd = Cmd(["sbatch", "-p", partition, "--gres=gpu:1", "--mem=12G", "--job-name=$job_name", script, args_str...])
    Base.run(cmd)
end
