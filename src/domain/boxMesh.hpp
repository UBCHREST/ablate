#ifndef ABLATELIBRARY_BOXMESH_HPP
#define ABLATELIBRARY_BOXMESH_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "domain.hpp"

namespace ablate::domain {
/**
 * Create a simple box mesh (1,2,3) dimension
 *
 * When used with the dm_plex_separate_marker each boundary "marker" or "Face Sets" as
 * 1D:
 *  x- left = 1
 *  x+ right =2
 * 2D:
 *  y+ top    = 3
 *  y- bottom = 1
 *  x+ right  = 2
 *  x- left   = 4
 * 3D:
 * y- bottom = 1
 * y+ top    = 2
 * z+ front  = 3
 * z- back   = 4
 * x+ right  = 5
 * x- left   = 6
 *
 */
class BoxMesh : public Domain {
   private:
    static DM CreateBoxDM(const std::string& name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary = {}, bool simplex = true);

   public:
    BoxMesh(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, std::vector<int> faces,
            std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary = {}, bool simplex = true, std::shared_ptr<parameters::Parameters> options = {});

    ~BoxMesh() override;
};
}  // namespace ablate::domain

#endif  // ABLATELIBRARY_BOXMESH_HPP
