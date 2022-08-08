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

    /**
     * call this function on root and wait to complete
     * @param comm
     */
    static void Once(MPI_Comm comm, std::function<void()>, int root = 0);

   private:
    MpiUtilities() = delete;
};
}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_PETSCUTILITIES_HPP
