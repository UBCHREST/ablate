#ifndef ABLATELIBRARY_BOXMESHBOUNDARYCELLS_HPP
#define ABLATELIBRARY_BOXMESHBOUNDARYCELLS_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "domain.hpp"

namespace ablate::domain {
class BoxMeshBoundaryCells : public Domain {
   private:
    static DM CreateBoxDM(std::string name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, bool simplex = true);

    static std::vector<std::shared_ptr<modifiers::Modifier>> AddBoundaryModifiers(std::vector<double> lower, std::vector<double> upper,  std::shared_ptr<domain::Region> mainRegion, std::shared_ptr<domain::Region> boundaryFaceRegion,
                                                                                 std::shared_ptr<domain::Region> boundaryCellRegion, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers);

   public:
    BoxMeshBoundaryCells(std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, std::vector<int> faces,
                         std::vector<double> lower, std::vector<double> upper, std::shared_ptr<domain::Region> mainRegion, std::shared_ptr<domain::Region> boundaryFaceRegion,
                         std::shared_ptr<domain::Region> boundaryCellRegion, bool simplex = true);

    ~BoxMeshBoundaryCells();
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_BOXMESH_HPP
