#include "hdf5OutputParticle.hpp"
#include <petsc.h>
#include <petscviewerhdf5.h>
#include "generators.hpp"
#include "monitors/runEnvironment.hpp"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

void ablate::monitors::particles::Hdf5OutputParticle::Register(std::shared_ptr<ablate::particles::Particles> particlesIn) {
    // store the flow
    particles = particlesIn;

    // build the file name
    outputFilePath = monitors::RunEnvironment::Get().GetOutputDirectory() / (particles->GetName() + extension);

    // setup the petsc viewer
    PetscViewerHDF5Open(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;
    DMSetOutputSequenceNumber(particles->GetParticleData()->dm, 0, 0) >> checkError;

    auto& particleData = particles->GetParticleData();
    ParticleView(particleData, petscViewer) >> checkError;
}

PetscErrorCode ablate::monitors::particles::Hdf5OutputParticle::OutputParticles(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::particles::Hdf5OutputParticle*)mctx;
    PetscErrorCode ierr = DMSetOutputSequenceNumber(monitor->particles->GetParticleData()->dm, steps, time);
    CHKERRQ(ierr);

    auto& particleData = monitor->particles->GetParticleData();
    ierr = ParticleView(particleData, monitor->petscViewer);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::particles::ParticleMonitor, ablate::monitors::particles::Hdf5OutputParticle, "outputs the particles and particle variables to an hdf5 file");
