#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include "mesh.hpp"

namespace ablate::mesh {
class BoxMesh : public Mesh {
   public:
    BoxMesh(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments, int dimensions);
};
}  // namespace ablate::mesh

#endif  // ABLATELIBRARY_BOXMESH_HPP
