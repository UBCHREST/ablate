#include "particleAverage.hpp"
#include <monitors/logs/stdOut.hpp>

ablate::monitors::ParticleAverage::ParticleAverage(int interval, std::shared_ptr<logs::Log> logIn) : interval(interval), log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}
void ablate::monitors::ParticleAverage::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    particles = std::dynamic_pointer_cast<particles::Particles>(monitorableObject);
    if (!particles) {
        throw std::invalid_argument("The ParticleAverage monitor can only be used with ablate::particles::Particles");
    }
}
PetscErrorCode ablate::monitors::ParticleAverage::OutputParticleAverage(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto monitor = (ablate::monitors::ParticleAverage *)mctx;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // if this is the first time step init the log
        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(PetscObjectComm((PetscObject)ts));
        }

        // Get the particle sizes
        PetscInt localParticleCount;
        PetscInt globalParticleCount;
        ierr = DMSwarmGetLocalSize(monitor->particles->GetParticleDM(), &localParticleCount);
        CHKERRQ(ierr);
        ierr = DMSwarmGetSize(monitor->particles->GetParticleDM(), &globalParticleCount);
        CHKERRQ(ierr);

        // compute the average particle location
        const PetscReal *coords;
        PetscInt dims;
        PetscReal avg[3] = {0.0, 0.0, 0.0};
        ierr = DMSwarmGetField(monitor->particles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
        CHKERRQ(ierr);
        for (PetscInt p = 0; p < localParticleCount; p++) {
            for (PetscInt n = 0; n < dims; n++) {
                avg[n] += coords[p * dims + n] / PetscReal(globalParticleCount);
            }
        }
        ierr = DMSwarmRestoreField(monitor->particles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
        CHKERRQ(ierr);

        // sum across all ranks
        PetscReal globAvg[3] = {0.0, 0.0, 0.0};
        PetscMPIInt bufferSize;
        ierr = PetscMPIIntCast(dims, &bufferSize);
        CHKERRQ(ierr);

        int mpiErr = MPI_Reduce(avg, globAvg, bufferSize, MPIU_REAL, MPI_SUM, 0, PetscObjectComm((PetscObject)monitor->particles->GetParticleDM()));
        CHKERRMPI(mpiErr);

        // print to the log
        monitor->log->Printf("%s ", monitor->particles->GetId().c_str());
        monitor->log->Print("Avg", dims, globAvg);
        monitor->log->Print("\n");
    }
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::ParticleAverage, "Outputs the average particle location in the domain", OPT(int, "interval", "output interval"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
