#ifndef ABLATELIBRARY_PARTICLECOUNT_HPP
#define ABLATELIBRARY_PARTICLECOUNT_HPP

#include <petsc.h>
#include <monitors/logs/log.hpp>
#include <particles/particles.hpp>

namespace ablate::monitors {

class ParticleCount : public Monitor {
   private:
    const PetscInt interval;
    std::shared_ptr<ablate::particles::Particles> particles;
    const std::shared_ptr<logs::Log> log;

    static PetscErrorCode OutputParticleCount(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

   public:
    ParticleCount(int interval, std::shared_ptr<logs::Log> log = {});

    void Register(std::shared_ptr<Monitorable>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputParticleCount; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_PARTICLECOUNT_HPP
