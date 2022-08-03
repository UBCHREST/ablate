#ifndef ABLATELIBRARY_COMPLETELABEL_HPP
#define ABLATELIBRARY_COMPLETELABEL_HPP

#include <domain/region.hpp>
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Wrapper for [DMPlexLabelComplete](https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexLabelComplete.html).  Complete the labels; such that if your label includes all faces, all vertices
 * connected are also labeled.
 */
class CompleteLabel : public Modifier {
   private:
    // label to complete
    const std::shared_ptr<domain::Region> region;

   public:
    explicit CompleteLabel(std::shared_ptr<domain::Region>);

    void Modify(DM&) override;

    std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_CREATELABEL_HPP
