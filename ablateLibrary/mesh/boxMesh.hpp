#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "mesh.hpp"

namespace ablate::mesh {
class BoxMesh : public Mesh {
   private:
    // Petsc options specific to the mesh. These may be null by default
    PetscOptions petscOptions;

   public:
    BoxMesh(std::string name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary = {}, bool simplex = true,
            std::shared_ptr<parameters::Parameters> options = {});

    ~BoxMesh();
};
}  // namespace ablate::mesh

#endif  // ABLATELIBRARY_BOXMESH_HPP
