#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "domain.hpp"

namespace ablate::domain {
class BoxMesh : public Domain {
   private:
    static DM CreateBoxDM(std::string name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary = {}, bool simplex = true);

   public:
    BoxMesh(std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, std::vector<int> faces,
            std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary = {}, bool simplex = true);

    ~BoxMesh();
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_BOXMESH_HPP
