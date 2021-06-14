#ifndef ABLATELIBRARY_DMPLEX_HPP
#define ABLATELIBRARY_DMPLEX_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include "mesh.hpp"

namespace ablate::mesh {

class DMPlex : public Mesh{
   private:
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    DMPlex(std::string name = "dmplex", std::shared_ptr<parameters::Parameters> options = {});
    ~DMPlex();
};
}

#endif  // ABLATELIBRARY_DMPLEX_HPP
