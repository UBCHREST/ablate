#ifndef ABLATELIBRARY_FIELD_FUNCTION_DENSITY_EXTRA_VARIABLES_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_DENSITY_EXTRA_VARIABLES_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::finiteVolume::fieldFunctions {

class DensityExtraVariables : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> eulerFunction;
    const std::vector<std::shared_ptr<mathFunctions::MathFunction>> evFunctions;

    static PetscErrorCode ComputeDensityEvFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    explicit DensityExtraVariables(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState, std::vector<std::shared_ptr<mathFunctions::MathFunction>> evFunctions,
                                   std::shared_ptr<ablate::domain::Region> region = {}, std::string nonConservedFieldName = {});
};

}  // namespace ablate::finiteVolume::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
