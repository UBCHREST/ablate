---
layout: default
title: Developing on CCR
parent: Development Guides
nav_order: 10
---

The University at Buffalo Center for Computational Research [(CCR)](http://www.buffalo.edu/ccr.html) is UB's Supercomputing center. The following are useful links to get started at CCR:

* [CCR Getting Started](http://www.buffalo.edu/ccr/support/getting-started.html)
* [CCR Knowledge Base](https://ubccr.freshdesk.com/support/home)
* [CCR OnDemand](https://ondemand.ccr.buffalo.edu) - an integrated, single access point for all of your HPC resources
* [CCR Coldfront](https://coldfront.ccr.buffalo.edu) - resource allocation management tool built

## Notes:

1. For the most recent development issues, notes, and workarounds for building PETSc or ABLATE on CCR check the [ABLATE repo wiki](https://github.com/UBCHREST/ablate/wiki).

1. Loading pre-built CHREST modules may also load any additional modules necessary to use the given software. This may change any preloaded modules you may have.

1. You must have read-access to the CHREST project directory to use the all CHREST pre-built modules. If you do not have access to the project directory it will be necessary to either use the libraries generally available on CCR or build the libraries yourself and install in your home directory.

1. To make CHREST pre-built libraries available during every login add
  ```bash
    module use /projects/academic/chrest/modules
  ```
  to your .bash_profile. Alternatively you can use the command in the terminal to make the modules available temporarily in that terminal.

1. The default for all CHREST modules is the debug version.

1. Most pre-build CHREST modules are indexed as ```version_day-month-year_GitCommit```, which refers to the version of the software, the date it was compiled, and the (short) Git commit hash of the compiled software.

## Available CHREST modules
The following are available CHREST modules:

1. PETSc

1. Ablate

1. petscXdmfGenerator

## Loading all CHREST modules
Loading all available CHREST modules can be done via
```bash
  module load chrest/release
```
for the release version or
```bash
  module load chrest/debug
```
for the debug version.

## Loading pre-built ABLATE
To load pre-build versions of Ablate in the terminal enter
  ```bash
    module load ablate/release
  ```
  for the release version or enter
  ```bash
  module load ablate/debug
  ```
  for the debug version. These will load all required modules for Ablate, add the ```ABLATE_DIR``` environment variable, and add ```ABLATE_DIR``` to ```path```.

  All available versions of ABLATE can be seen by using
  ```bash
  module spider ablate
  ```
  in the terminal window.


## Loading pre-built PETSc via CHREST modules
To compile ABLATE against pre-built PETSc modules enter in the terminal

  ```bash
    module load petsc-chrest/release
  ```

  for the release version of PETSc or enter

  ```bash
    module load petsc-chrest/debug
  ```

  for the debug version of PETSc. These will load all required modules for PETSc and add the ```PETSC_DIR``` and ```PETSC_ARCH``` environment variables.

  All available versions of PETSc pre-compiled for use with ABLATE can be seen by using

  ```bash
    module spider petsc-chrest
  ```

  in the terminal window.

## Building ABLATE on CCR
1. Clone and checkout the desired branch

    ```bash
      git clone git@github.com:UBCHREST/ablate.git
    ```
    or

    ```bash
      git clone https://github.com/UBCHREST/ablate.git
    ```

1. Load required PETSc modules -- See above

1. Create debug and release build directories

    ```bash
      mkdir ablate-debug
      mkdir ablate-release
    ```
1. Configure and build

    ```bash
      ## debug mode
      cmake -DCMAKE_BUILD_TYPE=Debug -B ablate-debug -S ablate
      make -C ablate-debug

      ## release
      cmake -DCMAKE_BUILD_TYPE=Release -B ablate-release -S ablate
      make -C ablate-release

    ```

## Submitting Jobs to CCR
CCR uses SLURM for scheduling and therefore job scripts specify the job. Details about [SLURM Commands](https://ubccr.freshdesk.com/support/solutions/articles/5000686927) and [Submitting a SLURM Job Script](https://ubccr.freshdesk.com/support/solutions/articles/5000688140-submitting-a-slurm-job-script) are provided by CCR. The following example script runs all tests of ablate. Save the script in either ```ablate-debug``` or ```ablate-release``` to test the appropriate version. Submit the job via the terminal command
```bash
  sbatch ablateTest.sh
```

##### Script to run all framework tests (ablateTest.sh)

 ```bash
   #!/bin/sh
   #SBATCH --partition=general-compute --qos=general-compute
   #SBATCH --time=00:15:00
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=2
   ##SBATCH --constraint=IB
   #SBATCH --mem=3000
   #SBATCH --job-name="ablate_framework_test"
   #SBATCH --output=ablate_framework_test-srun.out
   #SBATCH --mail-user=yourName@buffalo.edu
   #SBATCH --mail-type=ALL

   # Print the current environment
   echo "SLURM_JOBID="$SLURM_JOBID
   echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
   echo "SLURM_NNODES"=$SLURM_NNODES
   echo "SLURMTMPDIR="$SLURMTMPDIR

   echo "working directory = "$SLURM_SUBMIT_DIR

   # Load the required PETSc module. Change to specific version is necessary
   module load petsc-chrest/debug

   # The initial srun will trigger the SLURM prologue on the compute nodes.
   NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
   echo NPROCS=$NPROCS

   # The PMI library is necessary for srun
   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

   # Tell the tests what mpi command to use
   export TEST_MPI_COMMAND=srun

   # change to your build directory, either debug or release
   cd debug
   echo "current directory ="$PWD

   # Run all tests
   ctest

 ```
