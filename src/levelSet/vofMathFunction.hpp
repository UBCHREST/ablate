#ifndef ABLATELIBRARY_VOFMATHFUNCTION_HPP
#define ABLATELIBRARY_VOFMATHFUNCTION_HPP

#include <memory>
#include "domain/domain.hpp"
#include "mathFunctions/functionPointer.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::levelSet {
/**
 * Return the vertex level set values assuming a straight interface in the cell with a given normal vector.
 *
 * This class extends FunctionPointer to reduce duplicate code
 */
class VOFMathFunction : public mathFunctions::FunctionPointer {
   private:
    //! Hold a pointer to the domain to enable access to the cell information at a given point
    std::shared_ptr<ablate::domain::Domain> domain;

    //! function used to calculate the level set values at the vertices
    std::shared_ptr<ablate::mathFunctions::MathFunction> levelSet;

    /**
     * Static VOFMathFunctionPetscFunction that can be passed into petsc calls
     */
    static PetscErrorCode VOFMathFunctionPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    /**
     * Return the vertex level set values assuming a straight interface in the cell with a given normal vector.
     * @param domain Hold a pointer to the domain to enable access to the cell information at a given point
     * @param levelSet function used to calculate the level set values at the vertices
     */
    VOFMathFunction(std::shared_ptr<ablate::domain::Domain> domain, std::shared_ptr<ablate::mathFunctions::MathFunction> levelSet);
};
}  // namespace ablate::levelSet
#endif  // ABLATELIBRARY_VOFMATHFUNCTION_HPP
