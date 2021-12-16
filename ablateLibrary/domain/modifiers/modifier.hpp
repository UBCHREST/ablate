#ifndef ABLATELIBRARY_MODIFIER_HPP
#define ABLATELIBRARY_MODIFIER_HPP

#include <petsc.h>

namespace ablate::domain::modifiers {
class Modifier {
   public:
    virtual ~Modifier() = default;

    virtual void Modify(DM&) = 0;
};
}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MODIFIER_HPP
