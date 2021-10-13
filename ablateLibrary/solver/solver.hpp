#ifndef ABLATELIBRARY_SOLVER_HPP
#define ABLATELIBRARY_SOLVER_HPP

#include <vector>
#include "fieldDescriptor.hpp"

namespace ablate::flow {

class Solver {
   protected:
    // descriptions to the fields for this solver
    const std::vector<FieldDescriptor> flowFieldDescriptors;

    const std::string name;

    Solver(std::vector<FieldDescriptor> flowFieldDescriptors);

    // Petsc options specific to this solver. These may be null by default
    PetscOptions petscOptions;

   public:

};

}
#endif  // ABLATELIBRARY_SOLVER_HPP
