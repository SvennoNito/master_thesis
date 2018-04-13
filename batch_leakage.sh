#!/bin/bash
#SBATCH --partition=carl.p
#SBATCH --ntasks=48
#SBATCH --time=0-4
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=xxx@web.de
module load Python/3.5.2
hosts=$(srun hostname)
python -m scoop --hosts $hosts --prolog $(pwd)/load_module.sh -vv EA_cython.py -s -3 -n 1000 -g 100 -c 0 0 1 0 -t 0 10 20 30 40 50 60 70 80 90 100 -p EL gL -f voltage_base steady_state_voltage_stimend
