#!/bin/bash
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --job-name='chem_sens'
#SBATCH --output=out_chem_sens-%j.out
#SBATCH --error=error_chem_sens-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

#SBATCH --mail-user=danialfa@buffalo.edu
#SBATCH --mail-type=ALL


module use /projects/academic/chrest/modules
module load chrest/release


module load intel/19.5 && \
module load intel-mpi/2019.5 && \
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so && \
module load mkl/2019.5 && \
source $MKL/bin/mklvars.sh intel64 && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/academic/danialfa/software/dakota/dependencies/gsl/lib && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/academic/danialfa/software/dakota/dependencies/boost/1.49/lib && \
export DAK_INSTALL=/projects/academic/danialfa/software/dakota && \
export PATH=$DAK_INSTALL/bin:$DAK_INSTALL/share/dakota/test:$PATH && \
export PYTHONPATH=$DAK_INSTALL/share/dakota/Python:$PYTHONPATH

srun dakota -i sensitivity.in

