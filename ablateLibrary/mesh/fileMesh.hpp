#ifndef ABLATELIBRARY_FILEMESH_HPP
#define ABLATELIBRARY_FILEMESH_HPP

#include <filesystem>
#include "mesh.hpp"
#include <parameters/parameters.hpp>

namespace ablate::mesh {

class FileMesh : public Mesh {
   private:
    const std::filesystem::path path;
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    explicit FileMesh(std::string nameIn, std::filesystem::path path, std::shared_ptr<parameters::Parameters> options = {} );
    ~FileMesh() override;

};
};
#endif  // ABLATELIBRARY_FILEMESH_HPP
