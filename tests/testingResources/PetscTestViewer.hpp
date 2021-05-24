#ifndef petsctestviewer_h
#define petsctestviewer_h
#include <gtest/gtest.h>
#include <petscsys.h>
#include <cstdio>
#include <cstdlib>

namespace testingResources {

class PetscTestViewer{
   private:
    PetscViewer viewer;
    std::FILE* file;

   public:
    PetscTestViewer(MPI_Comm comm = PETSC_COMM_WORLD);
    ~PetscTestViewer();

    PetscViewer GetViewer() const{
        return viewer;
    }

    std::string GetString();
};

};  // namespace testingResources
#endif