#ifndef ABLATELIBRARY_LABELSUPPORT_HPP
#define ABLATELIBRARY_LABELSUPPORT_HPP

#include <petsc.h>

namespace ablate::domain::modifiers {

class LabelSupport {
   protected:
    void DistributeLabel(DM dm, DMLabel label);
};
}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_LABELSUPPORT_HPP
