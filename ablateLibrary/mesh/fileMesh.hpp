#ifndef ABLATELIBRARY_FILEMESH_HPP
#define ABLATELIBRARY_FILEMESH_HPP

#include <filesystem>
#include <parameters/parameters.hpp>
#include "mesh.hpp"

namespace ablate::mesh {

class FileMesh : public Mesh {
   private:
    const std::filesystem::path path;
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    explicit FileMesh(std::string nameIn, std::filesystem::path path, std::shared_ptr<parameters::Parameters> options = {});
    ~FileMesh() override;
};
};      // namespace ablate::mesh
#endif  // ABLATELIBRARY_FILEMESH_HPP
