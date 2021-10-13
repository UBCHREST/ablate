#!/usr/bin/python3

# Makes the LMOD file
def MakeModFile(PROJECT_DIR, ABLATE_ARCH, ABLATE_DIR, DEBUG, AVX512):
  import os

  # First get the modules that were loaded during compilation.
  MODS = os.getenv('LOADEDMODULES')
  MODS = MODS.split(':')

  modDir = PROJECT_DIR+"/modules/ablate/"
  modFile = modDir+ABLATE_ARCH+".lua"

  f = open(modFile, "w")

  f.write('whatis([[ablate_'+ABLATE_ARCH+']])\n')
  f.write('\n')
  if DEBUG==1:
    f.write('help([[Pre-compiled release version of Ablate.]])\n')
  else:
    f.write('help([[Pre-compiled debug version of Ablate.]])\n')
  f.write('\n')
  f.write('family("ablate")\n')
  f.write('\n')

  # Any module with salac or chrest in the name don't need to be included as prereqs
  for m in MODS:
    if m.find('salac')==-1 and m.find('petscXdmf')==-1 and m.find('chrest')==-1 and m.find('gcc')==-1:
      f.write('prereq("'+m+'")\n')

  f.write('\n')
  f.write('setenv("ABLATE_DIR", "'+ABLATE_DIR+'")\n')
  f.write('prepend_path("PATH", "'+ABLATE_DIR+'")\n')


  #f.write('project_dir = "'+PROJECT_DIR+'"')
  #f.write('petsc_dir = pathJoin(lib_dir, "petsc")\n')
  #f.write('petsc_bin = pathJoin(petsc_dir, "bin/")\n')
  #f.write('pkg_path = pathJoin(petsc_dir, "lib/pkgconfig/")\n')
  #f.write('\n')
  #f.write('setenv("PETSC_DIR", lib_dir)\n')
  #f.write('setenv("PETSC_ARCH", "petsc-'+PETSC_ARCH+'")\n')
  #f.write('prepend_path("PKG_CONFIG_PATH", pkg_path)\n')

  f.close()

  # Update the debug, release, and default lua files
  if DEBUG==1:
    symFile = modDir+"/debug.lua"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)
    symFile = modDir+"/default"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)
  else:
    if AVX512==1:
      symFile = modDir+"/release-avx512.lua"
    else:
      symFile = modDir+"/release.lua"
    if os.path.exists(symFile):
      os.remove(symFile)
    os.symlink(modFile, symFile)



# Returns the ABLATE_ARCH. Format will be day-month-year-commitID
def GetAblateArch(DEBUG, AVX512):
  from datetime import date
  #import re

  # Get the commit ID of ablate
  COMMIT_ID = os.popen('git rev-parse --short HEAD').read().rstrip()

  # Current date
  today = date.today()
  DATE = today.strftime("%d-%m-%Y")

  # Ablate version
  f = open('CMakeLists.txt');
  ABLATE_VERSION = ""
  searchStr = "ablateLibrary VERSION "
  for line in f:
    if not ABLATE_VERSION and searchStr in line:
      ABLATE_VERSION = line[line.find(searchStr)+len(searchStr):len(line)-2].strip()
  f.close()


  # The ABLATE_ARCH to use
  ABLATE_ARCH = "v" + ABLATE_VERSION + "_" + DATE + "_" + COMMIT_ID
  if(DEBUG==1):
    ABLATE_ARCH += '-debug'
  elif(AVX512==1):
    ABLATE_ARCH += '-avx512'

  return ABLATE_ARCH

if __name__ == '__main__':
  import sys
  import subprocess
  import os
  import multiprocessing
  from datetime import date

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

  # Make sure the petsc-version matches the requested compile type
  if "PETSC_ARCH" in os.environ:
    PETSC_ARCH = os.getenv('PETSC_ARCH')
    PETSC_DIR = os.getenv('PETSC_DIR')
    PETSC_DEBUG = ("debug" in PETSC_ARCH) or ("debug" in PETSC_DIR)
    PETSC_AVX512 = ("avx512" in PETSC_ARCH) or ("avx512" in PETSC_DIR)
    if PETSC_DEBUG and DEBUG==0:
      raise RuntimeError("Requested release version of ABLATE but debug version of PETSc is loaded.")
    elif not PETSC_DEBUG and DEBUG==1:
      raise RuntimeError("Requested debug version of ABLATE but release version of PETSc is loaded.")

    if PETSC_AVX512 and AVX512==0:
      raise RuntimeError("Requested release version of ABLATE with AVX512 support but that version is not PETSc is loaded.")
  else:
    raise RuntimeError("A compatible version of PETSc must be loaded.")

  ABLATE_ARCH = GetAblateArch(DEBUG, AVX512)

   # Where to install
  if "PROJECT_DIR" in os.environ:
    PROJECT_DIR = os.getenv('PROJECT_DIR')
  else:
    PROJECT_DIR = "/usr/workspace/ubchrest"

  ABLATE_DIR = PROJECT_DIR+'/lib/ablate/'+ABLATE_ARCH

  ## Now compile
  if not os.path.exists(ABLATE_DIR):
    os.makedirs(ABLATE_DIR,exist_ok=True) # Make the compilation directory

  cmd = ['cmake']
  if DEBUG==1:
    cmd.append('-DCMAKE_BUILD_TYPE=Debug')
  else:
    cmd.append('-DCMAKE_BUILD_TYPE=Release')

  cmd.append('-DCMAKE_C_COMPILER=mpicc')
  cmd.append('-DCMAKE_CXX_COMPILER=mpicxx')

  cmd.append('-B')
  cmd.append(ABLATE_DIR)

  retCode = subprocess.check_call(cmd)

  nprocs = multiprocessing.cpu_count()
  retCode = subprocess.check_call(['make', '-j', str(nprocs), '-C', ABLATE_DIR])


  # Make a new module file
  MakeModFile(PROJECT_DIR, ABLATE_ARCH, ABLATE_DIR, DEBUG, AVX512)


