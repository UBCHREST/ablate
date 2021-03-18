#ifndef ABLATELIBRARY_TESTINGAUXFIELDUPDATER_HPP
#define ABLATELIBRARY_TESTINGAUXFIELDUPDATER_HPP
#include <petsc.h>
#include <vector>
#include "petscTestingFunction.hpp"

namespace tests::ablateCore::support {

typedef PetscErrorCode (*SolutionFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

class TestingAuxFieldUpdater {
   private:
    std::vector<SolutionFunction> funcs;
    std::vector<void *> ctxs;

   public:
    void AddField(PetscTestingFunction &field) {
        funcs.push_back(PetscTestingFunction::ApplySolution);
        ctxs.push_back(&field);
    }
    static PetscErrorCode UpdateSourceTerms(TS ts, void *ctx);
};
}  // namespace tests::ablateCore::support

#endif  // ABLATELIBRARY_TESTINGAUXFIELDUPDATER_HPP
