#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include "mesh.hpp"

namespace ablate {
namespace mesh {
class BoxMesh : public Mesh {
   public:
    BoxMesh(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments, int dimensions);
};
}
}

#endif  // ABLATELIBRARY_BOXMESH_HPP
