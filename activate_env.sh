#!/bin/bash
# pwrk: Activate conda environment and run project build

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install or load it first."
    exit 1
fi

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate keypoint_factory
