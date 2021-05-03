#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include <vector>
#include "mesh.hpp"

namespace ablate::mesh {
class BoxMesh : public Mesh {
   public:
    BoxMesh(std::string name, std::map<std::string, std::string> arguments, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary,
            bool simplex);

   protected:
    inline static std::map<std::string, std::string> Merge(std::map<std::string, std::string> a, std::map<std::string, std::string> b) {
        a.merge(b);
        return a;
    }
};
}  // namespace ablate::mesh

#endif  // ABLATELIBRARY_BOXMESH_HPP
