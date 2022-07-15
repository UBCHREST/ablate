#ifndef ABLATELIBRARY_MPIUTILITIES_HPP
#define ABLATELIBRARY_MPIUTILITIES_HPP
#include <petsc.h>
#include <functional>

namespace ablate::utilities {

class MpiUtilities {
   public:
    /**
     * call to apply in function in order one by one (useful for setup)
     * @param comm
     */
    static void RoundRobin(MPI_Comm comm, std::function<void(int rank)>);

   private:
    MpiUtilities() = delete;
};
}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_PETSCUTILITIES_HPP
