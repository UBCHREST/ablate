#ifndef ABLATELIBRARY_TAGLABELINTERFACE_HPP
#define ABLATELIBRARY_TAGLABELINTERFACE_HPP

#include <memory>
#include "domain/region.hpp"
#include "labelSupport.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Class to label/tag all faces/cells on the interface between two labels.  The left/right designations are just used to separate the left/right labels.
 */
class TagLabelInterface : public Modifier, private LabelSupport {
   private:
    const std::shared_ptr<domain::Region> leftRegion;
    const std::shared_ptr<domain::Region> rightRegion;

    // the region to tag the boundary faces
    const std::shared_ptr<domain::Region> boundaryFaceRegion;

    // value of the boundary cell value
    const std::shared_ptr<domain::Region> leftBoundaryCellRegion;
    const std::shared_ptr<domain::Region> rightBoundaryCellRegion;

   public:
    explicit TagLabelInterface(std::shared_ptr<domain::Region> leftRegion, std::shared_ptr<domain::Region> rightRegion, std::shared_ptr<domain::Region> boundaryFaceRegion,
                               std::shared_ptr<domain::Region> leftBoundaryCellRegion = {}, std::shared_ptr<domain::Region> rightBoundaryCellRegion = {});

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::TagLabelInterface"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_TAGLABELBOUNDARY_HPP
