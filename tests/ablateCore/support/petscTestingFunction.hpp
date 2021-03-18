#ifndef ABLATELIBRARY_PETSCTESTINGFUNCTION_HPP
#define ABLATELIBRARY_PETSCTESTINGFUNCTION_HPP
#include <muParser.h>
#include <petsc.h>

namespace tests::ablateCore::support {
class PetscTestingFunction {
   private:
    double coordinate[3] = {0, 0, 0};
    double time = 0.0;
    mu::Parser parser;

   public:
    PetscTestingFunction(const PetscTestingFunction&) = delete;
    void operator=(const PetscTestingFunction&) = delete;
    explicit PetscTestingFunction(std::string formula);
    static PetscErrorCode ApplySolution(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    static PetscErrorCode ApplySolutionTimeDerivative(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx);
};
}  // namespace tests::ablateCore::support

#endif  // ABLATELIBRARY_PETSCTESTINGFUNCTION_HPP
