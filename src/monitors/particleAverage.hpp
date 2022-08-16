#ifndef ABLATELIBRARY_PARTICLEAVERAGE_HPP
#define ABLATELIBRARY_PARTICLEAVERAGE_HPP

#include <petsc.h>
#include <monitors/logs/log.hpp>
#include "particles/particleSolver.hpp"
#include "monitor.hpp"

namespace ablate::monitors {

class ParticleAverage : public Monitor {
   private:
    const PetscInt interval;
    std::shared_ptr<ablate::particles::ParticleSolver> particles;
    const std::shared_ptr<logs::Log> log;

    static PetscErrorCode OutputParticleAverage(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

   public:
    ParticleAverage(int interval, std::shared_ptr<logs::Log> log = {});

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputParticleAverage; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_PARTICLECOUNT_HPP
