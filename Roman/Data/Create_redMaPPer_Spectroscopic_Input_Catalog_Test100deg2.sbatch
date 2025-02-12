#!/bin/bash

#SBATCH --job-name=Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=caslake
#SBATCH --account=pi-chihway

source /project/chihway/setup/setup_basic.sh
python /home/ihsuan/Data/Create_redMaPPer_Spectroscopic_Input_Catalog_Test100deg2.py
