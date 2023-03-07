#ifndef ABLATELIBRARY_COLLAPSELABELS_HPP
#define ABLATELIBRARY_COLLAPSELABELS_HPP

#include <domain/region.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {
/**
 * Collapse all set values in a label to the provided value in each region. This also completes each label.
 */
class CollapseLabels : public Modifier {
   private:
    const std::vector<std::shared_ptr<domain::Region>> regions;

   public:
    CollapseLabels(std::vector<std::shared_ptr<domain::Region>> regions);

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::CollapseLabels"; }
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MERGELABELS_HPP
