#!/bin/bash
#SBATCH --partition=dp-dam
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH -n 1
#SBATCH -o cuda-out.%j
#SBATCH -e cuda-err.%j
#SBATCH --time=00:60:00
 
export PYTHON_EGG_CACHE=/p/project/joaiml/hpc_course/morris

ml CUDA
module load GCC/8.3.0 
module load ParaStationMPI/5.2.2-1
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load h5py/2.9.0-serial-Python-3.6.8 
module load scikit/2019a-Python-3.6.8

ml Intel ParaStationMPI
ml CUDA
module load extoll
module load pscom/5.3.1-1 

srun --cpu_bind=none python /p/home/jusers/riedel1/deep/2019-HPC-Course-MLDL-Parts/CNN-cpu.py

