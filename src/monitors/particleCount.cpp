#include "particleCount.hpp"
#include <monitors/logs/stdOut.hpp>
ablate::monitors::ParticleCount::ParticleCount(int interval, std::shared_ptr<logs::Log> logIn) : interval(interval), log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}
void ablate::monitors::ParticleCount::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    ablate::monitors::Monitor::Register(monitorableObject);

    particles = std::dynamic_pointer_cast<particles::ParticleSolver>(monitorableObject);
    if (!particles) {
        throw std::invalid_argument("The ParticleCount monitor can only be used with ablate::particles::Particles");
    }
}
PetscErrorCode ablate::monitors::ParticleCount::OutputParticleCount(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;


    auto monitor = (ablate::monitors::ParticleCount *)mctx;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // if this is the first time step init the log
        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(PetscObjectComm((PetscObject)ts));
        }

        PetscInt particleCount;
        PetscCall(DMSwarmGetSize(monitor->particles->GetParticleDM(), &particleCount));

        monitor->log->Printf("%s Count: %" PetscInt_FMT "\n", monitor->particles->GetSolverId().c_str(), particleCount);
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::ParticleCount, "Outputs the total number of particles in the domain", OPT(int, "interval", "output interval"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
