---
layout: default
title: Developing on CCR
parent: Development Guides
nav_order: 2
---

The University at Buffalo Center for Computational Research [(CCR)](http://www.buffalo.edu/ccr.html) is UB's Supercomputing center. The following are useful links to get started at CCR:

* [CCR Getting Started](http://www.buffalo.edu/ccr/support/getting-started.html)
* [CCR Knowledge Base](https://ubccr.freshdesk.com/support/home)
* [CCR OnDemand](https://ondemand.ccr.buffalo.edu) - an integrated, single access point for all of your HPC resources
* [CCR Coldfront](https://coldfront.ccr.buffalo.edu) - resource allocation management tool built

## Building PETSc on CCR
If a newer version of PETSc is required than those pre-built on CCR you will be required to build PETSC.

1. Download the desired version of PETSc into your desired location.
   ```bash
   git clone https://gitlab.com/petsc/petsc.git petsc
   ```
1. Load the required modules
   ```bash
   module load hdf5/1.12.0-mpi 
   module load gcc/10.2.0
   module load cmake/3.17.1
   ```
1. Inside the petsc folder, configure PETSc and build PETSc
   ```bash
       ./configure  \
        --with-mpi-dir=${I_MPI_ROOT}/intel64/ \
        --with-hdf5-dir=${HDF5} --download-ctetgen --download-fftw  \
 	    --download-egads --download-metis \
 	    --download-ml --download-mumps --download-netcdf --download-p4est \
 	    --download-parmetis --download-pnetcdf --download-scalapack \
 	    --download-slepc --download-suitesparse --download-superlu_dist \
 	    --download-triangle --with-zlib --with-libpng
   
      #  --with-debugging=0 can be added to build PETSc in release
   
      # Follow the on screen instructions to build and test
   ```
1. Make note of the ```PETSC_DIR``` and ```PETSC_ARCH``` reported in the previous step.

## Building ABLATE on CCR
1. Clone and checkout the desired branch
    ```bash
    git clone —recursive url/to/repo/or/fork
    ```
1. Load required modules
   - If loading prebuilt PETSc module
       ```bash
       module load cmake/3.17.1
       module load petsc/v3.14.0
     
       export PKG_CONFIG_PATH=${PETSC_DIR}/lib/pkgconfig
       ```
   - If loading from custom built PETSc
       ```bash
       module load hdf5/1.12.0-mpi 
       module load gcc/10.2.0
       module load cmake/3.17.1
     
       export PETSC_DIR="path/to/petsc/dir"
       export PETSC_ARCH=petsc_arch
       export PKG_CONFIG_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig

       ```
1. Create debug and release build directories
    ```bash
    mkdir debug
    mkdir release
    ```
1. Configure and build
    ```bash

    ## debug mode
    cmake -DCMAKE_BUILD_TYPE=Debug -B debug -S framework
    make -C debug

    # release
    cmake -DCMAKE_BUILD_TYPE=Release -B release -S framework
    make -C release
    ```
 
## Submitting Jobs to CCR
CCR uses SLURMS for scheduling and therefore job scripts specifying the job. Details about [SLURM Commands](https://ubccr.freshdesk.com/support/solutions/articles/5000686927) and [Submitting a SLURM Job Script](https://ubccr.freshdesk.com/support/solutions/articles/5000688140-submitting-a-slurm-job-script) are provided by CCR. The following example script runs all tests of the debug build of the ablate.

##### Script to run all framework tests (test.sbatch)
 ```bash
 #!/bin/sh
 #SBATCH --partition=general-compute --qos=general-compute
 #SBATCH --time=00:15:00
 #SBATCH --nodes=2
 #SBATCH --ntasks-per-node=2
 ##SBATCH --constraint=IB
 #SBATCH --mem=3000
 #SBATCH --job-name="chrest_framework_test"
 #SBATCH --output=chrest_framework_test-srun.out
 #SBATCH --mail-user=mtmcgurn@buffalo.edu
 #SBATCH --mail-type=ALL

 # Print the current environment
 echo "SLURM_JOBID="$SLURM_JOBID
 echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
 echo "SLURM_NNODES"=$SLURM_NNODES
 echo "SLURMTMPDIR="$SLURMTMPDIR

 echo "working directory = "$SLURM_SUBMIT_DIR

 # Load the required modules
 # Comment out if build with custom petsc
 module load cmake/3.17.1
 module load petsc/v3.14.0
 module list
 
 # Uncomment and set paths if built with custom petsc
#module load hdf5/1.12.0-mpi 
#module load gcc/10.2.0
#module load cmake/3.17.1

 # The initial srun will trigger the SLURM prologue on the compute nodes.
 NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
 echo NPROCS=$NPROCS

 # The PMI library is necessary for srun
 export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

 # Tell the tests what mpi command to use
 export TEST_MPI_COMMAND=srun

 # change to your build directory
 cd debug
 echo "current directory ="$PWD

 # Run all tests
 ctest
 ```