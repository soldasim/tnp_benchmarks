#!/bin/sh

module load Python/3.11
python -m pip install torch numpy matplotlib

julia scripts/run_bosip.jl "$@"
