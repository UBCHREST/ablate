#ifndef ABLATELIBRARY_TAGMESHBOUNDARYFACES_HPP
#define ABLATELIBRARY_TAGMESHBOUNDARYFACES_HPP

#include <memory>
#include "domain/region.hpp"
#include "labelSupport.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Mark/tag all faces on the boundary of the mesh
 */
class TagMeshBoundaryFaces : public Modifier, private LabelSupport {
   private:
    // the region to tag the boundary faces
    const std::shared_ptr<domain::Region> region;

   public:
    explicit TagMeshBoundaryFaces(std::shared_ptr<domain::Region> region);

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::TagMeshBoundaryFaces"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_TAGMESHBOUNDARYFACES_HPP
