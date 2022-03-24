#ifndef ABLATELIBRARY_radiate_HPP
#define ABLATELIBRARY_radiate_HPP

#include "radiationSolver.hpp"
#include "radiationSolver/radiationProcess.hpp"

namespace ablate::radiationSolver {

class radiate : public RadiationProcess {
   public:
    explicit radiate();

    void Initialize(ablate::radiationSolver::RadiationSolver& bSolver) override;

    static PetscErrorCode radiateFunction();
};

}  // namespace ablate::radiationSolver::lodi
#endif  // ABLATELIBRARY_radiate_HPP
