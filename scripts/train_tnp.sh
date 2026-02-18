#!/bin/sh

module load Python/3.11
python -m pip install torch numpy matplotlib

julia scripts/train_tnp.jl "$@"
