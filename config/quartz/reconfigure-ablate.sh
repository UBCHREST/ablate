#!/bin/bash

# Python doesn't let you run system commands in the current shell. To make loading the required modules easier create as bash script,
#   set the modules, and then run the reconfigure python script.

umask 027
shopt -s nocasematch

module purge
module load chrest/quartz

if [ -z "$1" ]
then
  echo "Loading Debug"
  module load petsc-chrest/debug
else
  if [[ $1 == "release" ]]
  then
    if [[ -z "$2" ]]
    then
      echo "Loading Release"
      module load petsc-chrest/release
    else
      if [[ $2 == 'avx512' ]]
      then
        echo "Loading Release AVX512"
        module load petsc-chrest/release-avx512
      else
        echo "Loading Release"
        module load petsc-chrest/release
      fi
    fi
  else
    echo "Loading Debug"
    module load petsc-chrest/debug
  fi
fi

module load gcc/10.2.1

module list

./reconfigure-ablate.py $1 $2
