#include "particleAverage.hpp"
#include <monitors/logs/stdOut.hpp>

ablate::monitors::ParticleAverage::ParticleAverage(int interval, std::shared_ptr<logs::Log> logIn) : interval(interval), log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}
void ablate::monitors::ParticleAverage::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    ablate::monitors::Monitor::Register(monitorableObject);

    particles = std::dynamic_pointer_cast<particles::ParticleSolver>(monitorableObject);
    if (!particles) {
        throw std::invalid_argument("The ParticleAverage monitor can only be used with ablate::particles::Particles");
    }
}
PetscErrorCode ablate::monitors::ParticleAverage::OutputParticleAverage(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;

    auto monitor = (ablate::monitors::ParticleAverage *)mctx;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // if this is the first time step init the log
        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(PetscObjectComm((PetscObject)ts));
        }

        // Get the particle sizes
        PetscInt localParticleCount;
        PetscInt globalParticleCount;
        PetscCall(DMSwarmGetLocalSize(monitor->particles->GetParticleDM(), &localParticleCount));
        PetscCall(DMSwarmGetSize(monitor->particles->GetParticleDM(), &globalParticleCount));

        // compute the average particle location
        const PetscReal *coords;
        PetscInt dims;
        PetscReal avg[3] = {0.0, 0.0, 0.0};
        PetscCall(DMSwarmGetField(monitor->particles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords));
        for (PetscInt p = 0; p < localParticleCount; p++) {
            for (PetscInt n = 0; n < dims; n++) {
                avg[n] += coords[p * dims + n] / PetscReal(globalParticleCount);
            }
        }
        PetscCall(DMSwarmRestoreField(monitor->particles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords));

        // sum across all ranks
        PetscReal globAvg[3] = {0.0, 0.0, 0.0};
        PetscMPIInt bufferSize;
        PetscCall(PetscMPIIntCast(dims, &bufferSize));

        int mpiErr = MPI_Reduce(avg, globAvg, bufferSize, MPIU_REAL, MPI_SUM, 0, PetscObjectComm((PetscObject)monitor->particles->GetParticleDM()));
        CHKERRMPI(mpiErr);

        // print to the log
        monitor->log->Printf("%s ", monitor->particles->GetSolverId().c_str());
        monitor->log->Print("Avg", dims, globAvg);
        monitor->log->Print("\n");
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::ParticleAverage, "Outputs the average particle location in the domain", OPT(int, "interval", "output interval"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
