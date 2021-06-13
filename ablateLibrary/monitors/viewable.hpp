#ifndef ABLATELIBRARY_VIEWABLE_HPP
#define ABLATELIBRARY_VIEWABLE_HPP
#include "monitorable.hpp"

namespace ablate::monitors {
class Viewable :public Monitorable {
   public:
    virtual void View(PetscViewer viewer, PetscInt steps, PetscReal time, Vec u) const = 0;
};
}

#endif  // ABLATELIBRARY_VIEWABLE_HPP
