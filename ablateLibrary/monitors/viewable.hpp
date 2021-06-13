#ifndef ABLATELIBRARY_VIEWABLE_HPP
#define ABLATELIBRARY_VIEWABLE_HPP

namespace ablate::monitors {
class Viewable {
   public:
    virtual const std::string& GetName() const =0;
    virtual void View(PetscViewer viewer, PetscInt steps, PetscReal time, Vec u) const = 0;
};
}

#endif  // ABLATELIBRARY_VIEWABLE_HPP
