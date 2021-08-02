#ifndef ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
#define ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>

namespace ablate::flow::fieldSolutions {

class Euler : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<mathFunctions::MathFunction> temperatureFunction;
    const std::shared_ptr<mathFunctions::MathFunction> pressureFunction;
    const std::shared_ptr<mathFunctions::MathFunction> velocityFunction;

    static PetscErrorCode EulerFromTemperatureAndPressure(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    Euler(std::shared_ptr<ablate::eos::EOS> eos, std::shared_ptr<mathFunctions::MathFunction> temperatureFunction, std::shared_ptr<mathFunctions::MathFunction> pressureFunction,
          std::shared_ptr<mathFunctions::MathFunction> velocityFunction);
    ~Euler() override = default;
};

}  // namespace ablate::flow::fieldSolutions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
