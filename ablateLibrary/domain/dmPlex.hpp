#ifndef ABLATELIBRARY_DMPLEX_HPP
#define ABLATELIBRARY_DMPLEX_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class DMPlex : public Domain {
   private:
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    DMPlex(std::string name = "dmplex", std::shared_ptr<parameters::Parameters> options = {});
    ~DMPlex() override;
};
}  // namespace ablate::mesh

#endif  // ABLATELIBRARY_DMPLEX_HPP
