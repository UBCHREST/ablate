#ifndef ABLATELIBRARY_TAGLABELBOUNDARY_HPP
#define ABLATELIBRARY_TAGLABELBOUNDARY_HPP

#include "domain/region.hpp"
#include <memory>
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Class to label/tag all faces on a label boundary
 */
class TagLabelBoundary : public Modifier {
   private:
    // the label to tag
    const std::shared_ptr<domain::Region> region;

    // the region to tag the boundary faces
    const std::shared_ptr<domain::Region> boundaryFaceRegion;

    // value of the boundary cell value
    const std::shared_ptr<domain::Region> boundaryCellRegion;

   public:
    explicit TagLabelBoundary(std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> boundaryFaceRegion, const std::shared_ptr<domain::Region> boundaryCellRegion = {});

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_TAGLABELBOUNDARY_HPP
