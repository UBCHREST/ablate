#ifndef ABLATELIBRARY_FILEMESH_HPP
#define ABLATELIBRARY_FILEMESH_HPP

#include <filesystem>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class FileMesh : public Domain {
   private:
    const std::filesystem::path path;
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    explicit FileMesh(std::string nameIn, std::filesystem::path path, std::shared_ptr<parameters::Parameters> options = {}, std::vector<std::shared_ptr<modifier::Modifier>> modifiers = {});
    ~FileMesh() override;
};
};      // namespace ablate::domain
#endif  // ABLATELIBRARY_FILEMESH_HPP
