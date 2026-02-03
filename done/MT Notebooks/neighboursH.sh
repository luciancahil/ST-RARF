#!/bin/sh

#SBATCH --time=128:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --job-name=H
#SBATCH --account=st-singha53-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=royhe@student.ubc.ca
#SBATCH --output=H.txt
#SBATCH --error=H_error.txt

bash run_H.sh
