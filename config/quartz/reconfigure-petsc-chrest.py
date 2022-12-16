#!/usr/bin/python3

# Makes the LMOD file
def MakeModFile(PROJECT_DIR, LIB_DIR, PETSC_ARCH, DEBUG, AVX512):
  import os

  # First get the modules that were loaded during compilation.
  MODS = os.getenv('LOADEDMODULES')
  MODS = MODS.split(':')

  modDir = PROJECT_DIR+"/modules/petsc-chrest/"
  modFile = modDir+PETSC_ARCH+".lua"

  PETSC_INSTALL_DIR = LIB_DIR+'/petsc/'+PETSC_ARCH

  f = open(modFile, "w")

  f.write('whatis([[petsc_chrest_'+PETSC_ARCH+']])\n')
  f.write('\n')
  if DEBUG==1:
    f.write('help([[Pre-compiled debug version of PETSc for Ablate.]])\n')
  else:
    if AVX512==1:
      f.write('help([[Pre-compiled release version with AVX512 support of PETSc for Ablate.]])\n')
    else:
      f.write('help([[Pre-compiled release version with SSE4.2 support of PETSc for Ablate.]])\n')

  f.write('\n')
  f.write('family("petsc_chrest")\n')
  f.write('\n')

  # Any module with ablate, salac, or chrest in the name don't need to be included as prereqs
  for m in MODS:
    if m.find('ablate')==-1 and m.find('salac')==-1 and m.find('chrest')==-1:
      f.write('prereq("'+m+'")\n')

  f.write('\n')
  #f.write('setenv("PETSC_DIR", "'+LIB_DIR+'/petsc/")\n')
  #f.write('setenv("PETSC_ARCH", "'+PETSC_ARCH+'")\n')
  f.write('setenv("PETSC_DIR", "'+PETSC_INSTALL_DIR+'")\n')
  f.write('setenv("PETSC_ARCH", "")\n')
  f.write('setenv("HDF5", "'+PETSC_INSTALL_DIR+'")\n')
  f.write('prepend_path("PKG_CONFIG_PATH", "'+PETSC_INSTALL_DIR+'/lib/pkgconfig/")\n')
  f.write('prepend_path("PATH", "'+PETSC_INSTALL_DIR+'/bin")\n')
  f.close()

  # Update the debug, release, and default lua files
  if DEBUG==1:
    symFile = modDir+"debug.lua"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)
    symFile = modDir+"default"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)
  else:
    if AVX512==1:
      symFile = modDir+"release-avx512.lua"
    else:
      symFile = modDir+"release.lua"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)

# Returns the PETSC_ARCH. Format will be day-month-year-commitID
def GetPetscArch(DEBUG, AVX512):
  from datetime import date
  #import re

  # Get the commit ID of petsc
  COMMIT_ID = os.popen('git rev-parse --short HEAD').read().rstrip()

  # Current date
  today = date.today()
  DATE = today.strftime("%d-%m-%Y")

  # Petsc version
  petsc_major = -1
  petsc_minor = -1
  petsc_subminor = -1
  f = open('./include/petscversion.h');
  for line in f:
    if petsc_major==-1 and "PETSC_VERSION_MAJOR" in line:
      petsc_major = int(line[line.find("PETSC_VERSION_MAJOR")+len("PETSC_VERSION_MAJOR"):].strip())
    if petsc_minor==-1 and "PETSC_VERSION_MINOR" in line:
      petsc_minor = int(line[line.find("PETSC_VERSION_MINOR")+len("PETSC_VERSION_MINOR"):].strip())
    if petsc_subminor==-1 and "PETSC_VERSION_SUBMINOR" in line:
      petsc_subminor = int(line[line.find("PETSC_VERSION_SUBMINOR")+len("PETSC_VERSION_SUBMINOR"):].strip())
  f.close()


  # The PETSC_ARCH to use
  PETSC_MAJOR = str(petsc_major)
  PETSC_MINOR = str(petsc_minor)
  PETSC_SUBMINOR = str(petsc_subminor)
  PETSC_ARCH = "v"+PETSC_MAJOR + "." + PETSC_MINOR + "." + PETSC_SUBMINOR + "_" + DATE + "_" + COMMIT_ID
  if(DEBUG==1):
    PETSC_ARCH += '-debug'
  elif(AVX512==1):
    PETSC_ARCH += '-avx512'

  return PETSC_ARCH


if __name__ == '__main__':
  import sys
  import subprocess
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure


  # Unset some system variables, if they are set
  if "PETSC_ARCH" in os.environ:
    os.environ.pop('PETSC_ARCH')
  #if "CC" in os.environ:
    #os.environ.pop('CC')
  #if "CXX" in os.environ:
    #os.environ.pop('CXX')
  #if "FC" in os.environ:
    #os.environ.pop('FC')

  # Make sure the PETSC_DIR points to the current directory
  os.environ['PETSC_DIR'] = os.getcwd()

  # Determine if this is debugging or not. Default is debugging
  # Also determine if this is avx512 (only for release version)
  DEBUG = 1
  AVX512 = 0;
  if(len(sys.argv)>1):
    if(sys.argv[1].lower()=="release"):
      DEBUG = 0
      if(len(sys.argv)>2):
        if(sys.argv[2].lower()=="avx512"):
          AVX512 = 1



    #if(int(sys.argv[1])>0):
      #DEBUG = 0

  # TChem has an issue with the config file on CCR. Fix it by replacing '\n'.join(args) with ' '.join(args)
#  os.system('perl -pi -e \'s/\\x27\\\\n\\x27.join/\\x27 \\x27.join/g\' ./config/BuildSystem/config/packages/tchem.py')


  ## Locations

  VAL_DIR = os.getenv('VAL_DIR')
  LIB_DIR = os.getenv('LIB_DIR')
  MKL_DIR = os.getenv('MKLROOT')

  if "PROJECT_DIR" in os.environ:
    PROJECT_DIR = os.getenv('PROJECT_DIR')
  else:
    PROJECT_DIR = "/usr/workspace/ubchrest"

  if "LIB_DIR" in os.environ:
    LIB_DIR = os.getenv('LIB_DIR')
  else:
    LIB_DIR=PROJECT_DIR+'/lib'

  # Petsc arch to use
  PETSC_ARCH = GetPetscArch(DEBUG, AVX512)

  if(DEBUG==1):
    coptflags = '-g -O0'
  else:
    if(AVX512==1):
      coptflags = '-g -O3 -xCORE-AVX512 -fp-model fast=2'
    else:
      coptflags = '-g -O3 -xAVX2 -fp-model fast=2'

  # Now configure PETSc
  configure_options = [
    '--COPTFLAGS='+coptflags,
    '--CXXOPTFLAGS='+coptflags,
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpif90',
    '--with-debugging='+str(DEBUG),
    '--download-ctetgen',
    '--download-tetgen',
    '--download-egads',
    '--download-metis',
    #'--download-ml',
    '--download-mumps',
    '--download-netcdf',
    '--download-p4est',
    '--download-parmetis',
    '--download-pnetcdf',
    '--download-scalapack',
    '--download-slepc',
    '--download-suitesparse',
    '--download-superlu_dist',
    '--download-kokkos',
    '--download-triangle',
    '--download-hdf5',
    '--with-valgrind-dir='+VAL_DIR,
    '--with-blaslapack-dir='+MKL_DIR,
    '--with-gdb',
    '--with-libpng',
    '--download-zlib',
    '--with-64-bit-indices=1',
    '--PETSC_ARCH='+PETSC_ARCH,
    '--prefix='+LIB_DIR+"/petsc/"+PETSC_ARCH,
  ]
  configure.petsc_configure(configure_options)

  # Make PETSc
  cmd = ['make']
  cmd.append('PETSC_DIR='+os.getcwd())
  cmd.append('PETSC_ARCH='+PETSC_ARCH)
  cmd.append('all');
  subprocess.run(cmd)

  #Install PETSc
  cmd.pop();
  cmd.append('install');
  subprocess.run(cmd)


  # Make a new module file
  MakeModFile(PROJECT_DIR, LIB_DIR, PETSC_ARCH, DEBUG, AVX512)


