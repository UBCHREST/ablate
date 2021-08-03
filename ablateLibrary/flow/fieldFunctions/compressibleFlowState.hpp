#ifndef ABLATELIBRARY_COMPRESSIBLEFLOWSTATE_HPP
#define ABLATELIBRARY_COMPRESSIBLEFLOWSTATE_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include <memory>
#include <vector>
namespace ablate::flow::fieldFunctions {

class CompressibleFlowState {
   private:
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<mathFunctions::MathFunction> temperatureFunction;
    const std::shared_ptr<mathFunctions::MathFunction> pressureFunction;
    const std::shared_ptr<mathFunctions::MathFunction> velocityFunction;

    std::vector<std::shared_ptr<mathFunctions::MathFunction>> massFractionFunctions;

   public:
    CompressibleFlowState(std::shared_ptr<ablate::eos::EOS> eos, std::shared_ptr<mathFunctions::MathFunction> temperatureFunction, std::shared_ptr<mathFunctions::MathFunction> pressureFunction,
                          std::shared_ptr<mathFunctions::MathFunction> velocityFunction, std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> yiFunctions = {});

    static PetscErrorCode ComputeEulerFromState(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
    static PetscErrorCode ComputeDensityYiFromState(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
};

}  // namespace ablate::flow::fieldFunctions
#endif  // ABLATELIBRARY_COMPRESSIBLEFLOWSTATE_HPP
