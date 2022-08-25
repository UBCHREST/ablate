#ifndef ABLATELIBRARY_DMPLEXCHECK_HPP
#define ABLATELIBRARY_DMPLEXCHECK_HPP

#include <domain/region.hpp>
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Call the DMPlexCheck petsc function
 */
class DMPlexCheck : public Modifier {
   public:
    void Modify(DM&) override;

    std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_CREATELABEL_HPP
