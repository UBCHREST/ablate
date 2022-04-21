#ifndef ABLATELIBRARY_MODIFIER_HPP
#define ABLATELIBRARY_MODIFIER_HPP

#include <petsc.h>
#include <iostream>

namespace ablate::domain::modifiers {
class Modifier {
   public:
    virtual ~Modifier() = default;

    virtual void Modify(DM&) = 0;

    virtual std::string ToString() const = 0;
};

std::ostream& operator<<(std::ostream& os, const Modifier& modifier);

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MODIFIER_HPP
