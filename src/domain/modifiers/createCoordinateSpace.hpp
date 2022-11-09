#ifndef ABLATELIBRARY_CREATECOORDINATESPACE_HPP
#define ABLATELIBRARY_CREATECOORDINATESPACE_HPP

#include <domain/region.hpp>
#include <memory>
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Wrapper for [DMPlexCreateCoordinateSpace](https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexLabelComplete.html).
 */
class CreateCoordinateSpace : public Modifier {
   private:
    //! degree of the finite element
    const PetscInt degree;

   public:
    explicit CreateCoordinateSpace(PetscInt degree);

    void Modify(DM&) override;

    std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_CREATECOORDINATESPACE_HPP
