---
layout: default
title: Developing on CCR
parent: Development Guides
---

The University at Buffalo Center for Computational Research [(CCR)](http://www.buffalo.edu/ccr.html) is UB's Supercomputing center. The following are useful links to get started at CCR:

* [CCR Getting Started](http://www.buffalo.edu/ccr/support/getting-started.html)
* [CCR Knowledge Base](https://ubccr.freshdesk.com/support/home)
* [CCR OnDemand](https://ondemand.ccr.buffalo.edu) - an integrated, single access point for all of your HPC resources
* [CCR Coldfront](https://coldfront.ccr.buffalo.edu) - resource allocation management tool built

## Building the Framework on CCR
1. Clone and checkout the desired branch
    ```bash
    git clone —recursive url/to/repo/or/fork
    ```
1. Load required modules
    ```bash
    module load cmake/3.13.4
    module load petsc/v3.14.0
    ```
1. Create debug and release build directories
    ```bash
    mkdir debug
    mkdir release
    ```
1. Configure and build
    ```bash
    export PKG_CONFIG_PATH=${PETSC_DIR}/lib/pkgconfig

    ## debug mode
    cmake -DCMAKE_BUILD_TYPE=Debug -B debug -S framework
    make -C debug

    # release
    cmake -DCMAKE_BUILD_TYPE=Release -B release -S framework
    make -C release
    ```
 
## Submitting Jobs to CCR
CCR uses SLURMS for scheduling and therefore job scripts specifying the job. Details about [SLURM Commands](https://ubccr.freshdesk.com/support/solutions/articles/5000686927) and [Submitting a SLURM Job Script](https://ubccr.freshdesk.com/support/solutions/articles/5000688140-submitting-a-slurm-job-script) are provided by CCR. The following example script runs all tests of the debug build of the framework.

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
 module load cmake/3.13.4
 module load petsc/v3.14.0
 module list

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